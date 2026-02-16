from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from models.safe_rl.policy.base_policy import Policy
from models.safe_rl.util.networks import MLPActor, EnsembleQCritic
from models.safe_rl.util.logger import EpochLogger
from models.safe_rl.util.torch_util import (count_vars, get_device_name, to_device, to_ndarray,
                                     to_tensor)
from torch.optim import Adam


class DDPG(Policy):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 actor_lr=5e-6,
                 critic_lr=3e-4,
                 ac_model="mlp",
                 hidden_sizes=[64, 64],
                 act_noise=0.1,
                 initial_noise=0.3,
                 decay_epoch=150,
                 use_noise_decay=True,
                 gamma=0.99,
                 polyak=0.995,
                 num_q=2,
                 safe=False,
                 **kwargs) -> None:
        r'''
        Deep Deterministic Policy Gradient (DDPG)

        Args:
            env (gym.Env): OpenAI Gym environment
            logger (EpochLogger): Logger for metrics and model saving
            actor_lr (float): Learning rate for policy optimization (default: 5e-6)
            critic_lr (float): Learning rate for Q-value learning (default: 3e-4)
            ac_model (str): Actor-critic model type, currently only "mlp" supported
            hidden_sizes (list): List of hidden layer sizes for networks (default: [64, 64])
            act_noise (float): Final noise scale for action exploration (default: 0.1)
            initial_noise (float): Initial noise scale for exploration (default: 0.3)
            decay_epoch (int): Number of epochs for noise decay (default: 150)
            use_noise_decay (bool): Whether to use noise decay (default: True)
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
        self.decay_epoch = decay_epoch
        self.use_noise_decay = use_noise_decay

        self.safe = safe

        ################ create actor critic model ###############
        self.obs_dim = env.observation_space.shape[-1]
        self.act_dim = env.action_space.shape[-1]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_lim = env.action_space.high.item()  # Extract scalar value
        self.act_lim_low = env.action_space.low.item()  # Extract scalar value
        
        if ac_model.lower() == "mlp":
            if isinstance(env.action_space, gym.spaces.Box):
                actor = MLPActor(self.obs_dim, self.act_dim, hidden_sizes, nn.ReLU,
                                 self.act_lim_low, self.act_lim)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                raise ValueError("Discrete action space does not support yet")
            critic = EnsembleQCritic(self.obs_dim,
                                     self.act_dim,
                                     hidden_sizes,
                                     nn.ReLU,
                                     num_q=num_q)
        else:
            raise ValueError(f"{ac_model} ac model does not support.")

        # Set up optimizer and target q models
        self._ac_training_setup(actor, critic)

        self.epoch = 0
        # Setup initial alpha, either using alpha decay or static value
        if self.use_noise_decay:
            self.initial_noise = initial_noise
            self.noise_end = act_noise
            # Define the decay function
            self.decay_func_noise = lambda x: self.noise_end + (self.initial_noise - self.noise_end) * np.exp(-5. * x / self.decay_epoch)
            # Set initial threshold using decay function
            self.act_noise = self.initial_noise
        else:
            self.act_noise = act_noise

        # Set up model saving
        self.save_model()

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.actor, self.critic])
        self.logger.log('\nNumber of parameters: \t actor pi: %d, \t critic q: %d, \n' %
                        var_counts)

    def _ac_training_setup(self, actor, critic):
        critic_targ = deepcopy(critic)
        actor_targ = deepcopy(actor)
        self.actor, self.actor_targ, self.critic, self.critic_targ = to_device(
            [actor, actor_targ, critic, critic_targ], get_device_name())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False
        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Select action based on observation. Adds exploration noise during training.

        Args:
            obs (np.ndarray): Current observation
            deterministic (bool): If True, return deterministic action (no exploration)
            with_logprob (bool): Unused, kept for API compatibility with SAC

        Returns:
            np.ndarray: Selected action
            None: Placeholder for compatibility with SAC API
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            a = self.actor_forward(self.actor, obs)
        
        # Convert to numpy and remove batch dimension
        a = np.squeeze(to_ndarray(a), axis=0)
        
        # Add exploration noise during training
        if not deterministic:
            noise = self.act_noise * np.random.randn(*a.shape) * self.act_lim
            a += noise
            
        # Clip action to valid range
        return np.clip(a, self.act_lim_low, self.act_lim), None

    def learn_on_batch(self, data: dict):
        '''
        Update policy and value functions using a batch of experience.

        Args:
            data (dict): Batch of data containing:
                - 'obs': Current observations
                - 'act': Actions taken
                - 'rew': Rewards received
                - 'obs2': Next observations
                - 'done': Episode termination flags
                - 'indicator': Safety indicators
        '''
        # Step 1: Update critic (Q-function)
        self._update_critic(data)

        # Step 2: Update actor (policy)
        # Freeze Q-networks to prevent wasted computation
        for p in self.critic.parameters():
            p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks for next update
        for p in self.critic.parameters():
            p.requires_grad = True

        # Step 3: Update target networks via polyak averaging
        self._polyak_update_target(self.critic, self.critic_targ, self.polyak)
        self._polyak_update_target(self.actor, self.actor_targ, self.polyak)

    def post_epoch_process(self):
        '''
        This is called once at the end of each epoch.
        Update noise (if use_noise_decay = True).
        '''
        self._update_noise()
        self.epoch += 1


    def _update_noise(self):
        if self.use_noise_decay:
            self.act_noise = self.decay_func_noise(self.epoch)

    def critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def actor_forward(self, actor, obs):
        '''
        Compute deterministic actions from the actor network.

        Args:
            actor (nn.Module): Actor network to use (either current or target)
            obs (torch.Tensor): Batch of observations [batch, obs_dim]

        Returns:
            torch.Tensor: Deterministic actions [batch, act_dim]
        '''
        return actor(obs)

    def _update_actor(self, data):
        '''
        Update the actor network using the deterministic policy gradient.
        Uses the current critic to evaluate action quality.
        '''
        def policy_loss():
            obs = to_tensor(data['obs'])
            # Get deterministic actions from current policy
            act = self.actor_forward(self.actor, obs)
            # Evaluate actions using current critic
            q_pi, _ = self.critic_forward(self.critic, obs, act)
            # Negative because we want to maximize Q-value
            return -q_pi.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        loss_pi = policy_loss()
        loss_pi.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=1)
        
        self.actor_optimizer.step()

        # Log metrics
        self.logger.store(LossPi=loss_pi.item())

    def _update_critic(self, data):
        '''
        Update Q-function parameters using the Bellman equation.
        Uses target networks for stability and an ensemble of Q-functions.
        '''
        def critic_loss():
            # Load and process batch data
            obs, act, reward, indicator, obs_next, done = (
                to_tensor(data['obs']), 
                to_tensor(data['act']), 
                to_tensor(data['rew']),
                to_tensor(data['indicator']),  
                to_tensor(data['obs2']), 
                to_tensor(data['done'])
            )

            # Include safety costs in reward if not using separate safety critic
            if not self.safe:
                reward = reward - (1 - indicator.to(reward.dtype))

            # Get current Q-values for all critics in ensemble
            _, q_list = self.critic_forward(self.critic, obs, act)
            
            # Compute target Q-values using target networks
            with torch.no_grad():
                # Get next actions from target policy
                act_targ_next = self.actor_forward(self.actor_targ, obs_next)
                # Compute target Q-values
                q_pi_targ, _ = self.critic_forward(self.critic_targ, obs_next, act_targ_next)
                # Bellman backup
                backup = reward + self.gamma * (1 - done) * q_pi_targ

            # Compute MSE loss for all critics in ensemble
            loss_q = self.critic.loss(backup, q_list)
            
            # Prepare logging info
            q_info = {f"QVals{i}": to_ndarray(q) for i, q in enumerate(q_list)}
            return loss_q, q_info
        
        # First run one gradient descent step for Q1 and Q2
        self.critic_optimizer.zero_grad()
        loss_critic, loss_q_info = critic_loss()
        loss_critic.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5, norm_type=1)

        self.critic_optimizer.step()

        self.logger.store(LossQ=loss_critic.item(), **loss_q_info)

    def _polyak_update_target(self, net, net_targ, polyak):
        '''
        Update target networks by polyak averaging.
        '''
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def save_model(self):
        self.logger.setup_pytorch_saver((self.actor, self.critic))

    def load_model(self, path):
        actor, critic = torch.load(path)
        self._ac_training_setup(actor, critic)