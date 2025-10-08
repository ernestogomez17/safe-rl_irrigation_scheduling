from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from safe_rl.policy import SAC, LagrangianPIDController
from safe_rl.policy.networks import EnsembleQCritic
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import (get_device_name, to_ndarray, to_tensor)
from torch.optim import Adam

class SACLagrangian(SAC):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 num_qc=2,
                 chance_constraint=0.95,
                 use_constraint_decay=True,
                 chance_start=0.65,
                 cost_critic_lr=1e-3,
                 KP_s_star=1.0,
                 KI_s_star=0.5,
                 KD_s_star=0.01,
                 KP_sfc=1.0,
                 KI_sfc=0.5,
                 KD_sfc=0.01,
                 KP_sw=5.0,
                 KI_sw=1.0,
                 KD_sw=0.1,
                 # Number of episodes to run for the multiplier update
                 lag_update_samples=200, 
                 safe=True,
                 optimistic_regularization=0.1,
                 use_regularization_decay=False,
                 temperature=1.0,  # Add temperature parameter
                 **kwargs) -> None:
        r'''
        Soft Actor-Critic with Lagrangian Constraints (SAC-Lagrangian)
        
        This implementation extends SAC to handle multiple safety constraints using
        Lagrangian multipliers with PID control. It supports both constant and
        decaying safety thresholds.

        Args:
            env (gym.Env): OpenAI Gym environment
            logger (EpochLogger): Logger for metrics and model saving
            num_qc (int): Number of safety critics in ensemble (default: 2)
            chance_constraint (float): Final safety threshold (default: 0.95)
            use_constraint_decay (bool): Whether to use threshold decay (default: True)
            chance_start (float): Initial safety threshold if using decay (default: 0.25)
            cost_critic_lr (float): Learning rate for safety critics (default: 5e-4)
            KP (float): Proportional gain for PID controller (default: 10)
            KI (float): Integral gain for PID controller (default: 1)
            KD (float): Derivative gain for PID controller (default: 0.0001)
            lag_update_samples (int): Number of episodes for Lagrange multiplier update (default: 200)
            per_state (bool): Whether to use per-state safety critics (default: True)
            safe (bool): Whether to use separate safety critics (default: True)
            **kwargs: Additional arguments passed to SAC parent class
        '''
        super().__init__(env, logger, **kwargs)
        self.env = env
        self.safe = safe
        self.use_constraint_decay = use_constraint_decay
        self.use_regularization_decay = use_regularization_decay

        if optimistic_regularization < 0 or optimistic_regularization > 1:
            raise ValueError("optimistic_regularization must be between 0 and 1")
        self.epoch = 0
        if self.use_regularization_decay:
            self.reg_start = optimistic_regularization
            self.decay_func_reg = lambda x: 0.0 + (self.reg_start - 0.0) * np.exp(-5. * x / self.decay_epoch)
            self.optimistic_regularization = self.reg_start
        else:
            self.optimistic_regularization = optimistic_regularization
        # Setup initial threshold values
        if self.use_constraint_decay:
            self.threshold_start = 1 - chance_start
            self.threshold_end = 1 - chance_constraint
            # Define the decay functions
            self.decay_func_threshold = lambda x: self.threshold_end + (self.threshold_start - self.threshold_end) * np.exp(-5. * x / self.decay_epoch)
            self.decay_func_sw = lambda x: 0.0 + (self.threshold_start - 0.0) * np.exp(-5. * x / self.decay_epoch)
            self.target_safety = self.threshold_start
            self.target_safety_sw = self.threshold_start
        else:
            self.target_safety = 1 - chance_constraint
            # The target for the SW constraint is 99% safety (relaxed from 100% as it's impossible with sigmoid)
            self.target_safety_sw = 0.0

        print("Target safety constraint: ", self.target_safety)

        self.num_qc = num_qc
        self.cost_critic_lr = cost_critic_lr
        self.lag_update_samples = lag_update_samples
        self.temperature = temperature
        
        # Instantiate separate critics for each cost type with temperature
        self.qc_s_star = EnsembleQCritic(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU, 
                                        num_q=self.num_qc, probabilistic=True, temperature=self.temperature)
        self.qc_sfc = EnsembleQCritic(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU, 
                                     num_q=self.num_qc, probabilistic=True, temperature=self.temperature)
        self.qc_sw = EnsembleQCritic(self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU, 
                                    num_q=self.num_qc, probabilistic=True, temperature=self.temperature)

        # Setup training for each critic
        self._qc_training_setup(self.qc_s_star, 's_star')
        self._qc_training_setup(self.qc_sfc, 'sfc')
        self._qc_training_setup(self.qc_sw, 'sw')

        # Instantiate separate controllers for each critic
        self.controller_s_star = LagrangianPIDController(KP_s_star, KI_s_star, KD_s_star)
        self.controller_sfc = LagrangianPIDController(KP_sfc, KI_sfc, KD_sfc)
        self.controller_sw = LagrangianPIDController(KP_sw, KI_sw, KD_sw)

    def _qc_training_setup(self, qc, prefix):
        '''
        Sets up a safety critic network and its target network.
        
        Args:
            qc (EnsembleQCritic): Safety critic network to set up
            prefix (str): Identifier for the safety critic (e.g., 's_star', 'sfc', 'sw')
        
        Creates for each critic:
            - Target network
            - Optimizer
            - Device placement
        '''
        # Move networks to appropriate device
        device = get_device_name()
        qc = qc.to(device)
        qc_targ = deepcopy(qc).to(device)

        # Set up network attributes and optimizer
        setattr(self, f'qc_{prefix}', qc)
        setattr(self, f'qc_{prefix}_targ', qc_targ)
        setattr(self, f'qc_{prefix}_optimizer', 
                Adam(qc.parameters(), lr=self.cost_critic_lr))

        # Freeze target network parameters
        for p in qc_targ.parameters():
            p.requires_grad = False

    def learn_on_batch(self, data: dict):
        self._update_critic(data)

        self._update_qc(data, self.qc_s_star, self.qc_s_star_targ, self.qc_s_star_optimizer, 'S_STAR')
        self._update_qc(data, self.qc_sfc, self.qc_sfc_targ, self.qc_sfc_optimizer, 'SFC')
        self._update_qc(data, self.qc_sw, self.qc_sw_targ, self.qc_sw_optimizer, 'SW')

        for p in self.critic.parameters(): p.requires_grad = False
        for qc in [self.qc_s_star, self.qc_sfc, self.qc_sw]:
            for p in qc.parameters(): p.requires_grad = False
        
        self._update_actor(data)

        for p in self.critic.parameters(): p.requires_grad = True
        for qc in [self.qc_s_star, self.qc_sfc, self.qc_sw]:
            for p in qc.parameters(): p.requires_grad = True

        self._polyak_update_target(self.critic, self.critic_targ, self.polyak)
        self._polyak_update_target(self.qc_s_star, self.qc_s_star_targ, self.polyak)
        self._polyak_update_target(self.qc_sfc, self.qc_sfc_targ, self.polyak)
        self._polyak_update_target(self.qc_sw, self.qc_sw_targ, self.polyak)

    def post_epoch_process(self):
        '''
        Performs end-of-epoch updates in the following order:
        1. Update Lagrange multipliers based on QC predictions (your approach)
        2. Update entropy coefficient alpha (if using decay or learning)
        3. Update safety thresholds (if using decay)
        4. Increment epoch counter
        '''
        # self._update_multipliers()   # Update Lagrange multipliers
        self._update_alpha()         # Update entropy coefficient
        self._update_threshold()     # Update safety thresholds
        self._update_regularization_term()
        self.epoch += 1

    def _update_regularization_term(self):
        if self.use_regularization_decay:
            self.optimistic_regularization = self.decay_func_reg(self.epoch)

    def _update_threshold(self):
        '''
        Update safety thresholds using exponential decay.
        
        Regular constraints decay from chance_start to chance_constraint.
        SW constraint decays from chance_start to 0.99 (near-perfect safety).
        
        The decay rate is -5/decay_epoch, providing smooth convergence
        to the final values over the training period.
        '''
        if self.use_constraint_decay:
            # Update regular safety constraints
            self.target_safety = self.decay_func_threshold(self.epoch)
            # Update sliding window safety constraint (needs stricter final value)
            self.target_safety_sw = self.decay_func_sw(self.epoch)

    def _update_actor(self, data):
        '''
        Update the actor network.
        '''
        def policy_loss():
            obs = to_tensor(data['obs'])
            act, logp_pi = self.actor_forward(obs, False, True)
            q_pi, _ = self.critic_forward(self.critic, obs, act)
            qc_pi_s_star, _ = self.critic_forward(self.qc_s_star, obs, act)
            qc_pi_sfc, _ = self.critic_forward(self.qc_sfc, obs, act)
            #qc_pi_chance, _ = self.critic_forward(self.qc_chance, obs, act)
            qc_pi_sw, _ = self.critic_forward(self.qc_sw, obs, act)

            # Critics predict P(safe), we want to maximize this probability
            # Convert to violation probabilities for constraint formulation
            predicted_violation_rate_s_star = 1 - qc_pi_s_star
            predicted_violation_rate_sfc = 1 - qc_pi_sfc
            predicted_violation_rate_sw = 1 - qc_pi_sw

            # Update multipliers based on batch-averaged QC predictions
            with torch.no_grad():
                # Average the batch predictions to get expected violation rates
                avg_violation_rate_s_star = predicted_violation_rate_s_star.mean().item()
                avg_violation_rate_sfc = predicted_violation_rate_sfc.mean().item()
                avg_violation_rate_sw = predicted_violation_rate_sw.mean().item()
                
                # Compute errors: how much we exceed the target violation rate
                error_s_star = avg_violation_rate_s_star - self.target_safety
                error_sfc = avg_violation_rate_sfc - self.target_safety
                error_sw = avg_violation_rate_sw - self.target_safety_sw

                # Update PID controllers with the averaged errors
                self.controller_s_star.update(error_s_star)
                self.controller_sfc.update(error_sfc)
                self.controller_sw.update(error_sw)

                # Get updated multipliers
                multiplier_s_star = self.controller_s_star.get_multiplier()
                multiplier_sfc = self.controller_sfc.get_multiplier()
                multiplier_sw = self.controller_sw.get_multiplier()

            # Entropy-regularized policy loss
            current_alpha = self.log_alpha.exp().item() if self.learn_alpha else self.alpha
            reward_loss = (current_alpha * logp_pi - q_pi).mean()

            # Safety objective: Lagrangian penalty
            # Only penalize when predicted violation exceeds target (built into multiplier value)
            safety_penalty_s_star = (multiplier_s_star * (predicted_violation_rate_s_star - self.target_safety)).mean()
            safety_penalty_sfc = (multiplier_sfc * (predicted_violation_rate_sfc - self.target_safety)).mean()
            safety_penalty_sw = (multiplier_sw * (predicted_violation_rate_sw - self.target_safety_sw)).mean()

            total_loss = reward_loss + safety_penalty_s_star + safety_penalty_sfc + safety_penalty_sw

            alpha_loss = None
            if self.learn_alpha:
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

            # Useful info for logging
            pi_info = dict(
                LagrangianSStar=to_ndarray(multiplier_s_star),
                LagrangianSFC=to_ndarray(multiplier_sfc),
                LagrangianSW=to_ndarray(multiplier_sw),
                QValuePolicy=to_ndarray(reward_loss),
                QcStarPrediction=to_ndarray(qc_pi_s_star.mean()),
                QcSFCPrediction=to_ndarray(qc_pi_sfc.mean()),
                QcSWPrediction=to_ndarray(qc_pi_sw.mean()),
                PredictedViolationSStar=to_ndarray(predicted_violation_rate_s_star.mean()),
                PredictedViolationSFC=to_ndarray(predicted_violation_rate_sfc.mean()),
                PredictedViolationSW=to_ndarray(predicted_violation_rate_sw.mean()),
                ErrorSStar=to_ndarray(error_s_star),
                ErrorSFC=to_ndarray(error_sfc),
                ErrorSW=to_ndarray(error_sw),
                SafetyPenaltySStar=to_ndarray(safety_penalty_s_star),
                SafetyPenaltySFC=to_ndarray(safety_penalty_sfc),
                SafetyPenaltySW=to_ndarray(safety_penalty_sw),
                Alpha=current_alpha # Log current alpha        
                )
            
            return total_loss, pi_info, alpha_loss

        self.actor_optimizer.zero_grad()
        total_loss, pi_info, alpha_loss = policy_loss()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=1)
        self.actor_optimizer.step()

        if self.learn_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Log actor update info
        self.logger.store(**pi_info, TotalLossActor=total_loss.item())

        if self.learn_alpha:
            self.logger.store(AlphaLoss=alpha_loss.item())

    # def _update_qc(self, data, qc, qc_targ, qc_optimizer, cost_key):
    #     def critic_loss():
    #         obs, act, cost, obs_next, done = (
    #             to_tensor(data['obs']),
    #             to_tensor(data['act']),
    #             to_tensor(data[cost_key]),
    #             to_tensor(data['obs2']),
    #             to_tensor(data['done'])
    #         )

    #         # Forward pass through the critic network
    #         _, q_list = self.critic_forward(qc, obs, act)
            
    #         # Bellman backup for Q functions
    #         with torch.no_grad():
    #             # Target actions come from *current* policy
    #             act_next, logp_a_next = self.actor_forward(obs_next,
    #                                                        deterministic=False,
    #                                                        with_logprob=True)
    #             # Target Q-values
    #             q_pi_targ, _ = self.critic_forward(qc_targ, obs_next, act_next)
    #             q_pi_targ = torch.clamp(q_pi_targ, min=0)
    #             backup = cost + self.gamma * (1 - done) * (q_pi_targ - self.alpha * logp_a_next)

    #         # MSE loss against Bellman backup
    #         loss_q, _ = qc.loss(backup, q_list)
            
    #         # Useful info for logging
    #         q_info = {"Loss" + cost_key: loss_q.item()}  # Using .item() to extract scalar value
    #         for i, q in enumerate(q_list):
    #             q_info[f"QCVals_{cost_key}_{i}"] = to_ndarray(q)

    #         return loss_q, q_info

    #     # Zero gradients before backward pass
    #     qc_optimizer.zero_grad()
    #     loss_qc, loss_qc_info = critic_loss()
    #     loss_qc.backward()  # Perform backward pass
        
    #     # Apply gradient clipping to prevent exploding gradients
    #     torch.nn.utils.clip_grad_norm_(qc.parameters(), max_norm=0.1, norm_type=1)

    #     qc_optimizer.step()  # Update critic parameters

    #     # Log critic update info, include QC thresholds if applicable
    #     self.logger.store(**loss_qc_info, QcThresChance=self.qc_thres_chance, QcThresSW=self.qc_thres_sw)

    def _update_qc(self, data, qc, qc_targ, qc_optimizer, key):
        '''
        Update safety critic parameters using safety indicators.
        '''
        def critic_loss():
            # Process batch data
            obs, act, is_safe, obs_next, done = (
                to_tensor(data['obs']),
                to_tensor(data['act']),
                to_tensor(data[key]),  # Binary safety indicator (1=safe, 0=unsafe)
                to_tensor(data['obs2']),
                to_tensor(data['done'])
            )

            # Get current safety predictions
            _, q_list = self.critic_forward(qc, obs, act)
            
            # Compute target values
            with torch.no_grad():
                # Sample next actions from current policy
                act_next, _ = self.actor_forward(obs_next,
                                               deterministic=False,
                                               with_logprob=False)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(qc_targ, obs_next, act_next)

                # Hybrid backup with optimistic regularization
                hybrid_q_targ = self.optimistic_regularization + (1 - self.optimistic_regularization) * q_pi_targ

                # The backup equation for probabilistic safety critics:
                # P(safe) = P(current_safe) * P(future_safe | current_safe)
                backup = is_safe * ((1 - done) * hybrid_q_targ + done)

            # Compute MSE loss
            loss_q = qc.loss(backup, q_list)
            
            # Prepare logging info
            q_info = {
                "LossQc_" + key: loss_q.item(),
                **{f"QcVals_{key}_{i}": to_ndarray(q) for i, q in enumerate(q_list)}
            }

            return loss_q, q_info

        # Zero gradients before backward pass
        qc_optimizer.zero_grad()
        loss_qc, loss_qc_info = critic_loss()
        loss_qc.backward()  # Perform backward pass
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(qc.parameters(), max_norm=0.5, norm_type=1)

        qc_optimizer.step()  # Update critic parameters

        # Log critic update info, include QC thresholds if applicable
        self.logger.store(**loss_qc_info, TargetSafety=self.target_safety, TargetSafetySW=self.target_safety_sw)

    def _update_multipliers(self):
        '''
        Update Lagrange multipliers using PID controllers.
        
        This method:
        1. Runs evaluation episodes to measure actual safety rates
        2. Computes safety violations for each constraint
        3. Updates PID controllers based on the violations
        4. Uses deterministic policy for reliable evaluation
        
        The update frequency is controlled by lag_update_samples (episodes per update).
        '''
        # Initialize safety statistics
        ep_is_safe_s_star_outcomes = []
        ep_is_safe_sfc_outcomes = []
        ep_is_safe_sw_outcomes = []

        # Switch to evaluation mode for deterministic actions
        self.actor.eval()

        try:
            for _ in range(self.lag_update_samples):
                obs, info = self.env.reset(options={'training_mode': True})  # Keep training mode True
                
                # Track episode safety - start with 1.0 and multiply with each step
                episode_was_safe_s_star = 1.0
                episode_was_safe_sfc = 1.0  
                episode_was_safe_sw = 1.0

                terminated, truncated = False, False
                step_count = 0
                max_steps = 1000  # Add safety limit to prevent infinite loops
                
                while not (terminated or truncated) and step_count < max_steps:
                    with torch.no_grad():
                        obs_tensor = to_tensor(obs).unsqueeze(0)
                        action, _ = self.actor_forward(obs_tensor,
                                                     deterministic=True,
                                                     with_logprob=False)
                    
                    obs, _, _, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
                    
                    # Get safety indicators for this step (these are probabilities/floats)
                    step_is_safe_s_star = info.get('S_STAR indicator', 1.0)
                    step_is_safe_sfc = info.get('SFC indicator', 1.0)
                    step_is_safe_sw = info.get('SW indicator', 1.0)

                    # Episode safety is the product of all step safety probabilities
                    episode_was_safe_s_star *= step_is_safe_s_star
                    episode_was_safe_sfc *= step_is_safe_sfc
                    episode_was_safe_sw *= step_is_safe_sw
                    
                    step_count += 1
                
                # Record episode outcomes
                ep_is_safe_s_star_outcomes.append(episode_was_safe_s_star)
                ep_is_safe_sfc_outcomes.append(episode_was_safe_sfc)
                ep_is_safe_sw_outcomes.append(episode_was_safe_sw)
        
        finally:
            # Ensure we return to training mode even if an exception occurs
            self.actor.train()
        
        # Calculate actual safety rates
        actual_safety_s_star = np.mean(ep_is_safe_s_star_outcomes)
        actual_safety_sfc = np.mean(ep_is_safe_sfc_outcomes)
        actual_safety_sw = np.mean(ep_is_safe_sw_outcomes)

        # Compute constraint violations for PID controllers
        actual_violation_rate_s_star = 1 - actual_safety_s_star
        actual_violation_rate_sfc = 1 - actual_safety_sfc  
        actual_violation_rate_sw = 1 - actual_safety_sw
        
        # Error > 0 means we're violating more than the target allows
        error_s_star = actual_violation_rate_s_star - self.target_safety
        error_sfc = actual_violation_rate_sfc - self.target_safety
        error_sw = actual_violation_rate_sw - self.target_safety_sw

        # Update PID controllers
        self.controller_s_star.update(error_s_star)
        self.controller_sfc.update(error_sfc)
        self.controller_sw.update(error_sw)

        # Log multiplier update info
        self.logger.store(
            ActualSafetySStar=actual_safety_s_star,
            ActualSafetySFC=actual_safety_sfc,
            ActualSafetySW=actual_safety_sw,
            ErrorSStar=error_s_star,
            ErrorSFC=error_sfc,
            ErrorSW=error_sw,
            MultiplierSStar=self.controller_s_star.get_multiplier(),
            MultiplierSFC=self.controller_sfc.get_multiplier(),
            MultiplierSW=self.controller_sw.get_multiplier(),
            EvaluationEpisodes=len(ep_is_safe_s_star_outcomes)
        )