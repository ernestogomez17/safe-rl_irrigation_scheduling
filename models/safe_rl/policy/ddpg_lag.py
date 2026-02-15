"""DDPG with PID-Lagrangian safety constraints (DDPG-Lagrangian).

Extends the deterministic policy gradient algorithm DDPG [1] with
multiple safety constraints enforced via PID-controlled Lagrange
multipliers [2].  Three soil-moisture constraints are supported:

* **S_STAR** (probabilistic) -- P(s_t >= S*) >= 1 - d
* **SFC**    (probabilistic) -- P(s_t <= SFC) >= 1 - d
* **SW**     (hard)          -- E[sum_t gamma^t c_t] <= 0,  c_t = 1{s_t < SW}

The probabilistic critics learn P(safe trajectory | s, a) via a
multiplicative Bellman backup with sigmoid outputs and BCE loss.  The SW
critic uses a standard discounted-cost value function (MSE loss, no
sigmoid).  Its raw output is normalized by the discounted episode horizon
H_gamma = (1 - gamma^T) / (1 - gamma) so that PID errors are in [0, 1],
commensurate with the probabilistic constraints.

References
----------
[1] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez,
    Y. Tassa, D. Silver, and D. Wierstra, "Continuous Control with
    Deep Reinforcement Learning," in Proc. Int. Conf. Learn. Repr.
    (ICLR), 2016.
[2] A. Stooke, J. Achiam, and P. Abbeel, "Responsive Safety in
    Reinforcement Learning by PID Lagrangian Methods," in Proc. 37th
    Int. Conf. Mach. Learn. (ICML), PMLR 119, 2020.
"""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.safe_rl.policy import DDPG, LagrangianPIDController
from models.safe_rl.util.networks import EnsembleQCritic
from models.safe_rl.util.logger import EpochLogger
from models.safe_rl.util.torch_util import get_device_name, to_ndarray, to_tensor


class DDPGLagrangian(DDPG):
    """DDPG with PID-Lagrangian safety constraints [1, 2].

    The actor loss augments the standard deterministic policy gradient
    objective with Lagrangian penalty terms -- one per constraint -- whose
    multipliers are adapted online by PID controllers following Stooke
    et al. (2020) [2].

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment (must expose ``n_days_ahead``).
    logger : EpochLogger
        Logging and model-checkpoint helper.
    num_qc : int
        Number of Q-networks in each safety-critic ensemble.
    chance_constraint : float
        Target safety probability for S_STAR / SFC (e.g. 0.95).
    use_constraint_decay : bool
        If True, anneal the probabilistic threshold from
        ``chance_start`` to ``chance_constraint`` over training.
    chance_start : float
        Initial (relaxed) safety probability when using decay.
    cost_critic_lr : float
        Learning rate for all safety critics.
    KP_s_star, KI_s_star, KD_s_star : float
        PID gains for the S_STAR Lagrange multiplier.
    KP_sfc, KI_sfc, KD_sfc : float
        PID gains for the SFC Lagrange multiplier.
    KP_sw, KI_sw, KD_sw : float
        PID gains for the SW Lagrange multiplier.
    safe : bool
        Whether to activate safety critics (set False for unconstrained).
    optimistic_regularization : float
        Pessimism offset epsilon in [0, 1] for the probabilistic backup.
    use_regularization_decay : bool
        If True, decay ``optimistic_regularization`` to 0 over training.
    temperature : float
        Sigmoid temperature for probabilistic critics.
    **kwargs
        Forwarded to :class:`DDPG` (gamma, hidden_sizes, lr, etc.).
    """

    def __init__(
        self,
        env: gym.Env,
        logger: EpochLogger,
        num_qc: int = 2,
        chance_constraint: float = 0.95,
        use_constraint_decay: bool = True,
        chance_start: float = 0.65,
        cost_critic_lr: float = 1e-3,
        KP_s_star: float = 1.0,
        KI_s_star: float = 0.5,
        KD_s_star: float = 0.01,
        KP_sfc: float = 1.0,
        KI_sfc: float = 0.5,
        KD_sfc: float = 0.01,
        KP_sw: float = 5.0,
        KI_sw: float = 1.0,
        KD_sw: float = 0.1,
        safe: bool = True,
        optimistic_regularization: float = 0.3,
        use_regularization_decay: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(env, logger, **kwargs)
        self.env = env
        self.safe = safe
        self.use_constraint_decay = use_constraint_decay
        self.use_regularization_decay = use_regularization_decay

        if not 0.0 <= optimistic_regularization <= 1.0:
            raise ValueError("optimistic_regularization must be in [0, 1]")

        self.epoch = 0

        # -- Optimistic regularization (epsilon in the probabilistic backup) --
        if self.use_regularization_decay:
            self.reg_start = optimistic_regularization
            self.decay_func_reg = lambda e: self.reg_start * np.exp(
                -5.0 * e / self.decay_epoch
            )
            self.optimistic_regularization = self.reg_start
        else:
            self.optimistic_regularization = optimistic_regularization

        # -- Probabilistic constraint thresholds (S_STAR, SFC) --
        if self.use_constraint_decay:
            self.threshold_start = 1 - chance_start
            self.threshold_end = 1 - chance_constraint
            self.decay_func_threshold = lambda e: (
                self.threshold_end
                + (self.threshold_start - self.threshold_end)
                * np.exp(-5.0 * e / self.decay_epoch)
            )
            self.target_safety = self.threshold_start
        else:
            self.target_safety = 1 - chance_constraint

        # -- Hard constraint threshold (SW) --
        # The SW critic estimates E[sum gamma^t c_t] with c_t in {0,1}.
        # The hard constraint is E[cost] <= 0, so the target is always 0.
        self.target_safety_sw = 0.0

        # Normalization horizon H_gamma so that error_sw lies in [0, 1]:
        #   H_gamma = sum_{t=0}^{T-1} gamma^t = (1 - gamma^T) / (1 - gamma)
        n_days_ahead = getattr(env, "n_days_ahead")
        episode_steps = int(np.ceil(365 / n_days_ahead))
        self.sw_cost_horizon = (1 - self.gamma ** episode_steps) / (1 - self.gamma)

        print(f"Target safety constraint: {self.target_safety}")
        print(
            f"SW cost horizon (normalization): {self.sw_cost_horizon:.2f}  "
            f"(T={episode_steps}, gamma={self.gamma})"
        )

        # -- Safety critics --
        self.num_qc = num_qc
        self.cost_critic_lr = cost_critic_lr
        self.temperature = temperature

        # S_STAR and SFC: probabilistic (sigmoid + BCE)
        self.qc_s_star = EnsembleQCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU,
            num_q=self.num_qc, probabilistic=True, temperature=self.temperature,
        )
        self.qc_sfc = EnsembleQCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU,
            num_q=self.num_qc, probabilistic=True, temperature=self.temperature,
        )
        # SW: standard value function (no sigmoid, MSE loss)
        self.qc_sw = EnsembleQCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, nn.ReLU,
            num_q=self.num_qc, probabilistic=False,
        )

        self._qc_training_setup(self.qc_s_star, "s_star")
        self._qc_training_setup(self.qc_sfc, "sfc")
        self._qc_training_setup(self.qc_sw, "sw")

        # -- PID Lagrangian controllers (Stooke et al., 2020) [2] --
        self.controller_s_star = LagrangianPIDController(KP_s_star, KI_s_star, KD_s_star)
        self.controller_sfc = LagrangianPIDController(KP_sfc, KI_sfc, KD_sfc)
        self.controller_sw = LagrangianPIDController(KP_sw, KI_sw, KD_sw)

        # Re-register saver now that safety critics exist
        self.save_model()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _qc_training_setup(self, qc: EnsembleQCritic, prefix: str) -> None:
        """Place *qc* on device, create a frozen target copy, and an optimizer."""
        device = get_device_name()
        qc = qc.to(device)
        qc_targ = deepcopy(qc).to(device)

        setattr(self, f"qc_{prefix}", qc)
        setattr(self, f"qc_{prefix}_targ", qc_targ)
        setattr(self, f"qc_{prefix}_optimizer",
                Adam(qc.parameters(), lr=self.cost_critic_lr))

        for p in qc_targ.parameters():
            p.requires_grad = False

    def save_model(self) -> None:
        """Checkpoint actor, reward critic, and all safety critics."""
        if hasattr(self, "qc_s_star"):
            self.logger.setup_pytorch_saver((
                self.actor, self.critic,
                self.qc_s_star, self.qc_sfc, self.qc_sw,
            ))
        else:
            # Called from parent __init__ before safety critics exist
            super().save_model()

    # ------------------------------------------------------------------
    # Per-gradient-step updates
    # ------------------------------------------------------------------

    def learn_on_batch(self, data: dict) -> None:
        """Single gradient step on a batch from the replay buffer.

        Update order: reward critic -> safety critics -> actor -> targets.
        """
        # 1. Reward critic
        self._update_critic(data)

        # 2. Safety critics
        self._update_qc(data, self.qc_s_star, self.qc_s_star_targ,
                         self.qc_s_star_optimizer, "S_STAR")
        self._update_qc(data, self.qc_sfc, self.qc_sfc_targ,
                         self.qc_sfc_optimizer, "SFC")
        self._update_qc(data, self.qc_sw, self.qc_sw_targ,
                         self.qc_sw_optimizer, "SW")

        # 3. Actor (freeze critics to save compute)
        for p in self.critic.parameters():
            p.requires_grad = False
        for qc in [self.qc_s_star, self.qc_sfc, self.qc_sw]:
            for p in qc.parameters():
                p.requires_grad = False

        self._update_actor(data)

        for p in self.critic.parameters():
            p.requires_grad = True
        for qc in [self.qc_s_star, self.qc_sfc, self.qc_sw]:
            for p in qc.parameters():
                p.requires_grad = True

        # 4. Polyak-averaged target networks
        self._polyak_update_target(self.critic, self.critic_targ, self.polyak)
        self._polyak_update_target(self.actor, self.actor_targ, self.polyak)
        self._polyak_update_target(self.qc_s_star, self.qc_s_star_targ, self.polyak)
        self._polyak_update_target(self.qc_sfc, self.qc_sfc_targ, self.polyak)
        self._polyak_update_target(self.qc_sw, self.qc_sw_targ, self.polyak)

    def _update_actor(self, data: dict) -> None:
        """Compute and apply the actor gradient.

        The loss combines the deterministic policy gradient objective
        with Lagrangian penalty terms for each safety constraint.  PID
        multipliers are updated in-line (primal-dual) following [2].
        """
        def policy_loss():
            obs = to_tensor(data["obs"])
            act = self.actor_forward(self.actor, obs)

            # Reward Q-value
            q_pi, _ = self.critic_forward(self.critic, obs, act)

            # Safety Q-values
            qc_pi_s_star, _ = self.critic_forward(self.qc_s_star, obs, act)
            qc_pi_sfc, _ = self.critic_forward(self.qc_sfc, obs, act)
            qc_pi_sw, _ = self.critic_forward(self.qc_sw, obs, act)

            # Probabilistic critics output P(safe); convert to P(violation)
            pred_viol_s_star = 1 - qc_pi_s_star
            pred_viol_sfc = 1 - qc_pi_sfc

            # SW critic outputs E[discounted cost]; normalize by horizon
            normalized_qc_sw = qc_pi_sw / self.sw_cost_horizon

            # --- PID multiplier update (no gradients needed) ---
            with torch.no_grad():
                error_s_star = pred_viol_s_star.mean().item() - self.target_safety
                error_sfc = pred_viol_sfc.mean().item() - self.target_safety
                error_sw = normalized_qc_sw.mean().item()

                self.controller_s_star.update(error_s_star)
                self.controller_sfc.update(error_sfc)
                self.controller_sw.update(error_sw)

                lam_s_star = self.controller_s_star.get_multiplier()
                lam_sfc = self.controller_sfc.get_multiplier()
                lam_sw = self.controller_sw.get_multiplier()

            # --- Policy loss ---
            reward_loss = -q_pi.mean()

            penalty_s_star = (lam_s_star * (pred_viol_s_star - self.target_safety)).mean()
            penalty_sfc = (lam_sfc * (pred_viol_sfc - self.target_safety)).mean()
            penalty_sw = (lam_sw * normalized_qc_sw).mean()

            total_loss = reward_loss + penalty_s_star + penalty_sfc + penalty_sw

            # --- Logging ---
            pi_info = dict(
                LagrangianSStar=to_ndarray(lam_s_star),
                LagrangianSFC=to_ndarray(lam_sfc),
                LagrangianSW=to_ndarray(lam_sw),
                QValuePolicy=to_ndarray(reward_loss),
                QcStarPrediction=to_ndarray(qc_pi_s_star.mean()),
                QcSFCPrediction=to_ndarray(qc_pi_sfc.mean()),
                QcSWCost=to_ndarray(normalized_qc_sw.mean()),
                PredictedViolationSStar=to_ndarray(pred_viol_s_star.mean()),
                PredictedViolationSFC=to_ndarray(pred_viol_sfc.mean()),
                ErrorSStar=to_ndarray(error_s_star),
                ErrorSFC=to_ndarray(error_sfc),
                ErrorSW=to_ndarray(error_sw),
                SafetyPenaltySStar=to_ndarray(penalty_s_star),
                SafetyPenaltySFC=to_ndarray(penalty_sfc),
                SafetyPenaltySW=to_ndarray(penalty_sw),
            )
            return total_loss, pi_info

        self.actor_optimizer.zero_grad()
        total_loss, pi_info = policy_loss()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1, norm_type=1)
        self.actor_optimizer.step()

        self.logger.store(**pi_info, TotalLossActor=total_loss.item())

    def _update_qc(self, data, qc, qc_targ, qc_optimizer, key: str) -> None:
        """Update one safety critic via Bellman regression.

        * **Probabilistic** (S_STAR, SFC): multiplicative backup with BCE loss.
        * **Standard** (SW): additive cost backup (cost = 1 - is_safe) with MSE.
        """
        def critic_loss():
            obs = to_tensor(data["obs"])
            act = to_tensor(data["act"])
            is_safe = to_tensor(data[key])          # 1 = safe, 0 = unsafe
            obs_next = to_tensor(data["obs2"])
            done = to_tensor(data["done"])

            _, q_list = self.critic_forward(qc, obs, act)

            with torch.no_grad():
                # Target actions from the *target* actor (DDPG-style)
                act_next = self.actor_forward(self.actor_targ, obs_next)
                q_pi_targ, _ = self.critic_forward(qc_targ, obs_next, act_next)

                if qc.probabilistic:
                    # P(safe traj) = is_safe * [ (1-done)*Q_targ + done*1 ]
                    hybrid = (
                        self.optimistic_regularization
                        + (1 - self.optimistic_regularization) * q_pi_targ
                    )
                    backup = is_safe * ((1 - done) * hybrid + done)
                else:
                    # Q(s,a) = cost + gamma * (1-done) * Q_targ
                    cost = 1.0 - is_safe
                    backup = cost + self.gamma * (1 - done) * q_pi_targ

            loss_q = qc.loss(backup, q_list)

            q_info = {
                f"LossQc_{key}": loss_q.item(),
                **{f"QcVals_{key}_{i}": to_ndarray(q) for i, q in enumerate(q_list)},
            }
            return loss_q, q_info

        qc_optimizer.zero_grad()
        loss_qc, info = critic_loss()
        loss_qc.backward()
        torch.nn.utils.clip_grad_norm_(qc.parameters(), max_norm=0.5, norm_type=1)
        qc_optimizer.step()

        self.logger.store(
            **info,
            TargetSafety=self.target_safety,
            TargetSafetySW=self.target_safety_sw,
        )

    # ------------------------------------------------------------------
    # Per-epoch updates
    # ------------------------------------------------------------------

    def post_epoch_process(self) -> None:
        """End-of-epoch housekeeping: noise, thresholds, regularization."""
        self._update_noise()
        self._update_threshold()
        self._update_regularization_term()
        self.epoch += 1

    def _update_threshold(self) -> None:
        """Anneal probabilistic constraint thresholds (S_STAR, SFC only)."""
        if self.use_constraint_decay:
            self.target_safety = self.decay_func_threshold(self.epoch)

    def _update_regularization_term(self) -> None:
        """Anneal the optimistic regularization epsilon toward zero."""
        if self.use_regularization_decay:
            self.optimistic_regularization = self.decay_func_reg(self.epoch)
