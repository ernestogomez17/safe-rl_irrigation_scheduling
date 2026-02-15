# Import necessary libraries
from env.water_environment import WaterEnvironment

import time as _time
import torch
from tqdm import tqdm

from models.safe_rl.policy import SAC, DDPG, SACLagrangian, DDPGLagrangian
from models.safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from models.safe_rl.util.torch_util import export_device_env_variable, seed_torch
from models.safe_rl.worker.collector import Collector
from results.plots import plot_simulation_data


# ---------------------------------------------------------------------------
# Helper: build agent from config dicts
# ---------------------------------------------------------------------------

_AGENT_CLASSES = {
    'DDPG': DDPG,
    'DDPGLagrangian': DDPGLagrangian,
    'SAC': SAC,
    'SACLagrangian': SACLagrangian,
}

_LAGRANGIAN_KEYS = [
    'chance_constraint', 'cost_critic_lr',
    'KP_s_star', 'KI_s_star', 'KD_s_star',
    'KP_sfc',   'KI_sfc',   'KD_sfc',
    'KP_sw',    'KI_sw',    'KD_sw',
    'optimistic_regularization', 'temperature',
]


def _build_agent(model_type, env, logger, agent_params):
    """Construct the RL agent from *agent_params* without manual key extraction."""
    cls = _AGENT_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(_AGENT_CLASSES)}")

    # Common kwargs shared by every agent
    kwargs = dict(
        gamma=agent_params['gamma'],
        polyak=agent_params['polyak'],
        hidden_sizes=agent_params['hidden_sizes'],
        actor_lr=agent_params.get('actor_lr'),
        critic_lr=agent_params.get('critic_lr'),
    )

    # Lagrangian-specific kwargs
    if 'Lagrangian' in model_type:
        kwargs['chance_constraint'] = agent_params['chance_const']
        for key in _LAGRANGIAN_KEYS:
            if key == 'chance_constraint':
                continue  # already added above
            if key in agent_params:
                kwargs[key] = agent_params[key]

    return cls(env, logger, **kwargs)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_metrics(logger, epoch, total_steps, elapsed_sec=None, verbose=True):
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('TotalEnvInteracts', total_steps)
    for key in logger.logger_keys:
        logger.log_tabular(key, average_only=False)
    if elapsed_sec is not None:
        logger.log_tabular('Time', elapsed_sec)
    logger.dump_tabular(x_axis="TotalEnvInteracts", verbose=verbose)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_agent(env_params, agent_params, training_params, model_directory, identifier):
    # ---- Reproducibility & hardware ----
    seed = training_params.get('seed')
    seed_torch(seed)
    torch.set_num_threads(training_params.get('threads', 1))
    export_device_env_variable(
        training_params.get('device'),
        id=training_params.get('device_id'))

    # ---- Environment (no redundant reset — Collector will reset) ----
    env = WaterEnvironment(**env_params)

    # ---- Logger ----
    logger_kwargs = setup_logger_kwargs(identifier, data_dir=model_directory)
    logger = EpochLogger(**logger_kwargs)

    # ---- Agent ----
    model_type = agent_params['model_type']
    agent = _build_agent(model_type, env, logger, agent_params)

    model_name = model_type
    if 'Lagrangian' in model_name:
        model_name = model_name.replace('Lagrangian', '-PIDLagrangian')

    # ---- Collector (handles warmup internally) ----
    batch_size = training_params.get('batch_size')
    max_episode_steps = training_params.get('max_episode_steps')
    collector = Collector(
        env=env, policy=agent, logger=logger,
        batch_size=batch_size, timeout_steps=max_episode_steps)

    # ---- Training hyper-parameters ----
    epochs = training_params.get('num_epochs')
    sample_episode_num = training_params.get('sample_episode_num')
    episode_rerun_num = training_params.get('episode_rerun_num')
    eval_episode_num = training_params.get('evaluate_episode_num')
    save_interval = training_params.get('plot_save_frequency')

    # ---- Tracking arrays (one entry per epoch) ----
    all_rewards = []      # mean training reward per epoch
    all_violations = []   # mean training violations per epoch
    all_eval_rewards = []     # mean eval reward per epoch
    all_eval_violations = []  # mean eval violations per epoch
    all_eval_ep_lens = []     # mean eval episode length per epoch
    best_eval_violations = float('inf')
    best_eval_reward = float('-inf')

    total_steps = 0
    start_time = _time.time()

    for epoch in range(epochs):
        # ---- 1. Collect data ----
        steps, mean_reward, mean_violations = collector.work(
            sample_episode_num=sample_episode_num)
        total_steps += steps

        all_rewards.append(mean_reward)
        all_violations.append(mean_violations)

        # ---- 2. Gradient updates ----
        # Aim for ≈1 gradient step per new env transition (UTD ≈ 1).
        # The old formula (episode_rerun_num * steps // batch_size) could
        # yield very few updates when steps < batch_size.
        train_steps = max(episode_rerun_num * steps // batch_size, 1)
        for _ in tqdm(range(train_steps),
                      desc=f'Epoch {epoch + 1}/{epochs}',
                      leave=False):
            data = collector.get_sample()
            agent.learn_on_batch(data)

        # ---- 3. End-of-epoch hooks (Lagrange multipliers, alpha, etc.) ----
        if hasattr(agent, 'post_epoch_process'):
            agent.post_epoch_process()

        # ---- 4. Evaluation ----
        (history_It, history_Rain, history_st,
         history_ET_o, history_ETmax,
         history_Kc, history_rho) = collector.eval(
             eval_episode_num=eval_episode_num)

        # ---- 5. Check for best model (fewest eval violations) ----
        # Must read stats BEFORE _log_metrics, which calls dump_tabular
        # and clears epoch_dict.
        eval_stats = logger.get_stats('eval/TestEpNumViolations')
        mean_eval_violations = eval_stats[0]  # mean
        eval_reward_stats = logger.get_stats('eval/TestEpRet')
        mean_eval_reward = eval_reward_stats[0]
        eval_len_stats = logger.get_stats('eval/TestEpLen')
        mean_eval_ep_len = eval_len_stats[0]

        all_eval_rewards.append(mean_eval_reward)
        all_eval_violations.append(mean_eval_violations)
        all_eval_ep_lens.append(mean_eval_ep_len)

        _log_metrics(logger, epoch, total_steps,
                     _time.time() - start_time)

        if mean_eval_violations < best_eval_violations:
            best_eval_violations = mean_eval_violations
            best_eval_reward = mean_eval_reward
            agent.save_model()  # save best
            logger.save_state({'env': env}, None)

        # ---- 6. Periodic plots & checkpoint ----
        if (epoch % save_interval == 0) or (epoch == epochs - 1):
            plot_simulation_data(
                history_It, history_Rain, history_st,
                history_ET_o, history_ETmax, history_Kc, history_rho,
                env.S_STAR, env.SFC, env.SW,
                model_directory, epoch, model_name)

    # ---- Final save ----
    agent.save_model()

    # ---- Return metrics for hyper-parameter optimisation ----
    return {
        'all_rewards': all_rewards,
        'all_violations': all_violations,
        'all_eval_rewards': all_eval_rewards,
        'all_eval_violations': all_eval_violations,
        'all_eval_ep_lens': all_eval_ep_lens,
        'best_eval_violations': best_eval_violations,
        'best_eval_reward': best_eval_reward,
        'total_episodes': len(all_violations) * sample_episode_num,
    }