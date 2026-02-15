import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG = False  # Set to True for debugging loss computation

# "normal" or "uniform" or None
INIT_METHOD = "normal"

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Builds a multi-layer perceptron (MLP).
    The final layer's activation is determined by output_activation.
    """
    if INIT_METHOD == "normal":
        initializer = nn.init.xavier_normal_
    elif INIT_METHOD == "uniform":
        initializer = nn.init.xavier_uniform_
    else:
        initializer = None
    
    bias_init = 0.0
    layers = []

    # Loop until the second-to-last size to add hidden layers and activations
    for j in range(len(sizes) - 2):
        layer = nn.Linear(sizes[j], sizes[j+1])
        if initializer is not None:
            initializer(layer.weight)
            nn.init.constant_(layer.bias, bias_init)
        layers += [layer, activation()]
    
    # --- CORRECTED SECTION ---
    # Add the final layer AND the specified output activation function
    final_layer = nn.Linear(sizes[-2], sizes[-1])
    if initializer is not None:
        initializer(final_layer.weight)
        nn.init.constant_(final_layer.bias, bias_init)
    
    # Add the final linear layer, followed by its activation function
    layers += [final_layer, output_activation()]
    # --- END CORRECTION ---
    
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, action_low, action_high):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_base = mlp(pi_sizes, activation)
        self.pi_activation = nn.Tanh() # Tanh is now an explicit part of the actor
        self.action_high = action_high
        self.action_low = action_low

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        logits = self.pi_base(obs)
        tanh_output = self.pi_activation(logits) # Apply Tanh to the logits
        # Scale to action range
        scaled_action = self.action_low + 0.5 * (self.action_high - self.action_low) * (tanh_output + 1)
        return scaled_action

class SquashedGaussianMLPActor(nn.Module):
    '''
    Probabilistic actor, can also be used as a deterministic actor
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, LOG_STD_MAX=2, LOG_STD_MIN=-20):
        super().__init__()
        # The base network ends with the last hidden layer (linear output)
        self.net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN

    def forward(self,
                obs,
                deterministic=False,
                with_logprob=True,
                with_distribution=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2, probabilistic=False, temperature=1.0):
        """
        Ensemble Q-Critic with optional probabilistic outputs.
        
        Args:
            obs_dim: Observation space dimension
            act_dim: Action space dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            num_q: Number of Q-networks in ensemble
            probabilistic: Whether to use probabilistic (sigmoid) outputs
            temperature: Temperature parameter for sigmoid scaling when probabilistic=True
        """
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.probabilistic = probabilistic
        
        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
            for i in range(num_q)
        ])
        
        # Create temperature-scaled sigmoid if using probabilistic outputs
        if self.probabilistic:
            self.sigmoid_activation = BetterSigmoid(temperature=temperature)

    def forward(self, obs, act):
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        q_list = []
        
        for q_net in self.q_nets:
            q_raw = torch.squeeze(q_net(data), -1)
            
            if self.probabilistic:
                # Use BetterSigmoid for temperature-scaled probabilistic output
                q_prob = self.sigmoid_activation(q_raw)
                q_list.append(q_prob)
            else:
                q_list.append(q_raw)
        
        return q_list

    def predict(self, obs, act):
        """
        Mapping: (s, a) -> logits -> Q(s, a).
        
        Returns:
            Tuple of (min_q, q_list) where min_q is the minimum across ensemble
        """
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        """Compute loss for all Q-networks in ensemble.
        
        Uses BCE for probabilistic critics (outputs in [0,1]) and MSE for
        standard reward critics (unbounded outputs).
        """
        if self.probabilistic:
            # Binary cross-entropy is the correct loss for probabilistic outputs.
            # MSE has vanishing gradients near 0 and 1 when combined with sigmoid.
            eps = 1e-7
            losses = [F.binary_cross_entropy(
                q.clamp(eps, 1 - eps),
                target.clamp(eps, 1 - eps)
            ) for q in q_list]
        else:
            losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)

class ScaledTanh(nn.Module):
    """
    Activation function that scales the output of Tanh to the [0, 1] range.
    """
    def forward(self, x):
        return (torch.tanh(x) + 1) / 2

class BetterSigmoid(nn.Module):
    """
    A sigmoid activation with better gradient properties for safety critics.
    Uses a temperature parameter to control steepness.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x):
        return torch.sigmoid(x / self.temperature)