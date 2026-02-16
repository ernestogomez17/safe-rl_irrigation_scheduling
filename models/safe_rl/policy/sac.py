from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from models.safe_rl.policy.base_policy import Policy
from models.safe_rl.util.networks import SquashedGaussianMLPActor, EnsembleQCritic
from models.safe_rl.util.logger import EpochLogger
from models.safe_rl.util.torch_util import (count_vars, get_device_name, to_device,
                                     to_ndarray, to_tensor)
from torch.optim import Adam


class SAC(Policy):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 actor_lr=8e-5,
                 critic_lr=2e-4,
                 ac_model="mlp",
                 hidden_sizes=[64, 64],
                 alpha=0.1,
                 initial_alpha=0.3,
                 decay_epoch=150,
                 use_alpha_decay=True,
                 learn_alpha=False,
                 gamma=0.99,
                 polyak=0.995,
                 num_q=2,
                 safe=False,
                 **kwargs) -> None:
        r'''
        Soft Actor-Critic (SAC) with automatic entropy tuning and alpha decay options.
        
        Implementation based on the paper:
        "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning 
        with a Stochastic Actor" by Haarnoja et al.

        Args:
            env (gym.Env): OpenAI Gym environment
            logger (EpochLogger): Logger for metrics and model saving
            actor_lr (float): Learning rate for policy optimization (default: 1e-5)
            critic_lr (float): Learning rate for Q-value learning (default: 1e-4)
            ac_model (str): Actor-critic model type, currently only "mlp" supported
            hidden_sizes (list): List of hidden layer sizes for networks (default: [64, 64])
            alpha (float): Final entropy regularization coefficient (default: 0.1)
            initial_alpha (float): Initial alpha value for decay (default: 0.3)
            decay_epoch (int): Number of epochs for alpha decay (default: 150)
            use_alpha_decay (bool): Whether to use alpha decay (default: True)
            learn_alpha (bool): Whether to automatically tune alpha (default: False)
            gamma (float): Discount factor (default: 0.99)
            polyak (float): Interpolation factor for polyak averaging (default: 0.995)
            num_q (int): Number of Q-functions in ensemble (default: 2)
            safe (bool): Whether to use safety critic (default: False)
        '''
        super().__init__()

        self.logger = logger
        self.gamma = gamma
        self.polyak = polyak
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_sizes = hidden_sizes
        self.safe = safe
        self.learn_alpha = learn_alpha
        self.use_alpha_decay = use_alpha_decay
        self.decay_epoch = decay_epoch
        self.device = get_device_name()

        ################ create actor critic model ###############
        self.obs_dim = env.observation_space.shape[-1] 
        self.act_dim = env.action_space.shape[-1]
        self.act_lim = env.action_space.high.item()
        self.act_lim_low = env.action_space.low.item()
        
        '''
        Notice: The output action are normalized in the range [-1, 1], 
        so please make sure your action space's high and low are suitable
        '''

        if ac_model.lower() == "mlp":
            actor = SquashedGaussianMLPActor(self.obs_dim, self.act_dim, hidden_sizes, nn.ReLU)
            critic = EnsembleQCritic(self.obs_dim, self.act_dim, hidden_sizes, nn.ReLU, num_q=num_q)
        else:
            raise ValueError(f"{ac_model} ac model does not support.")

        # Set up optimizer and target q models
        self._ac_training_setup(actor, critic)

        # Alpha
        if learn_alpha and use_alpha_decay:
            raise ValueError("Cannot use both alpha learning and alpha decay simultaneously!")

        self.epoch = 0
        if self.learn_alpha:
            self.target_entropy = -self.act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.actor_lr)
            self.alpha = self.log_alpha.exp().item()
        elif self.use_alpha_decay:
            self.initial_alpha = initial_alpha
            self.alpha_end = alpha
            # Define the decay function
            self.decay_func_alpha = lambda x: self.alpha_end + (self.initial_alpha - self.alpha_end) * np.exp(-5. * x / self.decay_epoch)
            # Set initial threshold using decay function
            self.alpha = self.initial_alpha
        else:
            self.alpha = alpha

        # Set up model saving
        self.save_model()

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.actor, self.critic])
        self.logger.log('\nNumber of parameters: \t actor pi: %d, \t critic q: %d, \n' % var_counts)

    def _ac_training_setup(self, actor, critic):
        critic_targ = deepcopy(critic)
        self.actor, self.critic, self.critic_targ = to_device(
            [actor, critic, critic_targ], self.device)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Select action based on observation using the stochastic policy.
        
        The action is sampled from a Gaussian distribution during training
        and uses the mean action during evaluation (deterministic=True).

        Args:
            obs (np.ndarray): Current observation
            deterministic (bool): If True, use mean action instead of sampling
            with_logprob (bool): Whether to return the log probability of the action

        Returns:
            np.ndarray: Selected action
            float or None: Log probability of the action (if with_logprob=True)
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            a, logp_a = self.actor_forward(obs, deterministic, with_logprob)
        # squeeze them to the right shape
        a, logp_a = np.squeeze(to_ndarray(a),
                               axis=0), np.squeeze(to_ndarray(logp_a))
        return a, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Update policy and value functions using a batch of experience.
        Implements the core SAC algorithm with entropy regularization.

        Args:
            data (dict): Batch of data containing:
                - 'obs': Current observations
                - 'act': Actions taken
                - 'rew': Rewards received
                - 'obs2': Next observations
                - 'done': Episode termination flags
                - 'indicator': Safety indicators
        '''
        self._update_critic(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ, self.polyak)
    
    def post_epoch_process(self):
        '''
        This is called once at the end of each epoch.
        Update alpha (if use_alpha_decay = True).
        '''
        self._update_alpha()
        self.epoch += 1

    def _update_alpha(self):
        if self.use_alpha_decay:
            self.alpha = self.decay_func_alpha(self.epoch)

    def critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def actor_forward(self, obs, deterministic=False, with_logprob=True):
        '''
        Compute actions using the stochastic policy network.
        Samples from a Gaussian distribution and applies tanh transformation.
        
        Args:
            obs (torch.Tensor): Batch of observations [batch, obs_dim]
            deterministic (bool): If True, return mean action instead of sampling
            with_logprob (bool): Whether to compute log probabilities

        Returns:
            torch.Tensor: Selected actions [batch, act_dim]
            torch.Tensor or None: Log probabilities of the actions
        '''
        a, logp = self.actor(obs, deterministic, with_logprob)

        # Scale action to [act_lim_low, act_lim]
        scaled_action = self.act_lim_low + 0.5 * (self.act_lim - self.act_lim_low) * (a + 1)

        return scaled_action, logp

    def _update_actor(self, data):
        '''
        Update the actor network using the reparameterization trick.
        Maximizes expected return plus entropy using the current Q-function.
        Also updates the entropy coefficient (alpha) if automatic tuning is enabled.
        '''
        def policy_loss():
            obs = to_tensor(data['obs'])
            act, logp_pi = self.actor_forward(obs, False, True)
            q_pi, q_list = self.critic_forward(self.critic, obs, act)

            # Entropy-regularized policy loss
            current_alpha = self.log_alpha.exp().item() if self.learn_alpha else self.alpha
            loss_pi = (current_alpha * logp_pi - q_pi).mean()

            alpha_loss = None
            if self.learn_alpha:
                # The goal is to make the policy's entropy match the target entropy.
                # We use .detach() on the entropy term so that this loss doesn't affect the actor weights.
                alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

            # Useful info for logging
            pi_info = dict(LogPi=to_ndarray(logp_pi.mean()))
            return loss_pi, pi_info, alpha_loss

        self.actor_optimizer.zero_grad()
        loss_pi, pi_info, alpha_loss = policy_loss()
        loss_pi.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=1)

        self.actor_optimizer.step()

        if self.learn_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Update the python-native alpha for logging and use in the next actor update
            self.alpha = self.log_alpha.exp().item()

        # Log actor update info
        self.logger.store(LossPi=loss_pi.item(), Alpha=self.alpha, **pi_info)
        if self.learn_alpha:
            self.logger.store(AlphaLoss=alpha_loss.item())
            
    def _update_critic(self, data):
        '''
        Update Q-function parameters using the Bellman equation.
        Uses an ensemble of critics and includes entropy in the target value.
        
        The target value is:
            reward + gamma * (1 - done) * (Q_target(next_obs, next_act) - alpha * log_prob)
        where next_act comes from the current policy.
        '''
        def critic_loss():
            obs, act, reward, indicator, obs_next, done = (
                to_tensor(data['obs']), 
                to_tensor(data['act']), 
                to_tensor(data['rew']),
                to_tensor(data['indicator']), 
                to_tensor(data['obs2']), 
                to_tensor(data['done'])
            )

            # Embed cost into reward if not using separate safety critic
            if not self.safe:
                reward = reward - (1 - indicator.to(reward.dtype))

            _, q_list = self.critic_forward(self.critic, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                act_next, logp_a_next = self.actor_forward(obs_next,
                                                           deterministic=False,
                                                           with_logprob=True)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next,
                                                   act_next)
                backup = reward + self.gamma * (1 - done) * (
                    q_pi_targ - self.alpha * logp_a_next)

            # MSE loss against Bellman backup
            loss_q = self.critic.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5, norm_type=1)

        self.critic_optimizer.step()

        self.logger.store(LossQ=loss_critic.item(), **loss_q_info)
        
    def _polyak_update_target(self, critic, critic_targ, polyak):
        '''
        Update target networks by polyak averaging.
        '''
        with torch.no_grad():
            for p, p_targ in zip(critic.parameters(),
                                 critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def save_model(self):
        self.logger.setup_pytorch_saver((self.actor, self.critic))

    def load_model(self, path):
        actor, critic = torch.load(path)
        self._ac_training_setup(actor, critic)