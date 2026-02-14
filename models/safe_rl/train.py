# Import necessary libraries
from data.water_environment import WaterEnvironment

import time
import torch
from tqdm import tqdm

from safe_rl.policy import SAC, DDPG, SACLagrangian, DDPGLagrangian
from safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from safe_rl.util.torch_util import export_device_env_variable, seed_torch
from safe_rl.worker.collector import Collector
from plots import plot_simulation_data, plot_results

def _log_metrics(logger, epoch, total_steps, time=None, verbose=True):
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('TotalEnvInteracts', total_steps)
    for key in logger.logger_keys:
        logger.log_tabular(key, average_only=False)
    if time is not None:
        logger.log_tabular('Time', time)
    # Dump the tabular data into the logger
    logger.dump_tabular(x_axis="TotalEnvInteracts", verbose=verbose)

def train_agent(env_params, agent_params, training_params, model_directory, identifier):
    seed = training_params.get('seed')
    device = training_params.get('device')
    device_id = training_params.get('device_id')
    threads = training_params.get('threads')
    
    seed_torch(seed)
    torch.set_num_threads(threads)
    export_device_env_variable(device, id=device_id)  

    env = WaterEnvironment(**env_params)
    env.reset()

    # Extract common parameters
    gamma = agent_params['gamma']
    polyak = agent_params['polyak']
    hidden_sizes = agent_params['hidden_sizes']
    actor_lr = agent_params.get('actor_lr')
    critic_lr = agent_params.get('critic_lr')
    
    # Extract Lagrangian-specific parameters only for Lagrangian models
    model_type = agent_params['model_type']
    is_lagrangian = 'Lagrangian' in model_type
    
    if is_lagrangian:
        chance_const = agent_params['chance_const']
        cost_critic_lr = agent_params.get('cost_critic_lr')
        kp_s_star = agent_params.get('KP_s_star')
        ki_s_star = agent_params.get('KI_s_star')
        kd_s_star = agent_params.get('KD_s_star')
        kp_sfc = agent_params.get('KP_sfc')
        ki_sfc = agent_params.get('KI_sfc')
        kd_sfc = agent_params.get('KD_sfc')
        kp_sw = agent_params.get('KP_sw')
        ki_sw = agent_params.get('KI_sw')
        kd_sw = agent_params.get('KD_sw')
        temperature = agent_params.get('temperature')
        optimistic_regularization = agent_params.get('optimistic_regularization')

    batch_size = training_params.get('batch_size')
    max_episode_steps = training_params.get('max_episode_steps')
    save_interval = training_params.get('plot_save_frequency')

    logger_kwargs = setup_logger_kwargs(identifier, data_dir=model_directory)
    logger = EpochLogger(**logger_kwargs)
    
    model_name = model_type
    if "Lagrangian" in model_name:
        model_name = model_name.replace("Lagrangian", "-PIDLagrangian")

    # Agent setup based on model type
    if model_type == 'DDPG':
        agent = DDPG(
            env, logger,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            polyak=polyak,
            hidden_sizes=hidden_sizes
        )

    elif model_type == 'DDPGLagrangian':
        agent = DDPGLagrangian(
            env, logger,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            cost_critic_lr=cost_critic_lr,
            polyak=polyak,
            hidden_sizes=hidden_sizes,
            chance_constraint=chance_const,
            KP_s_star=kp_s_star,
            KI_s_star=ki_s_star,
            KD_s_star=kd_s_star,
            KP_sfc=kp_sfc,
            KI_sfc=ki_sfc,
            KD_sfc=kd_sfc,
            KP_sw=kp_sw,
            KI_sw=ki_sw,
            KD_sw=kd_sw,
            optimistic_regularization=optimistic_regularization,
            temperature=temperature,
        )

    elif model_type == 'SAC':
        agent = SAC(
            env, logger,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            polyak=polyak,
            hidden_sizes=hidden_sizes
        )

    elif model_type == 'SACLagrangian':
        agent = SACLagrangian(
            env, logger,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            cost_critic_lr=cost_critic_lr,
            polyak=polyak,
            hidden_sizes=hidden_sizes,
            chance_constraint=chance_const,
            KP_s_star=kp_s_star,
            KI_s_star=ki_s_star,
            KD_s_star=kd_s_star,
            KP_sfc=kp_sfc,
            KI_sfc=ki_sfc,
            KD_sfc=kd_sfc,
            KP_sw=kp_sw,
            KI_sw=ki_sw,
            KD_sw=kd_sw,
            optimistic_regularization=optimistic_regularization,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    worker = Worker(env=env, policy=agent, logger=logger, batch_size=batch_size,
                             timeout_steps=max_episode_steps)
    epochs = training_params.get('num_epochs')

    total_steps = 0
    start_time = time.time()

    all_rewards = []
    all_violations = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_steps = 0

        steps, rewards, violations = worker.work(sample_episode_num=training_params['sample_episode_num'])
        all_rewards.append(rewards)
        all_violations.append(violations)
        epoch_steps += steps

        train_steps = training_params['episode_rerun_num'] * epoch_steps // batch_size
        for i in tqdm(range(train_steps), desc=f'Training Epoch {epoch + 1}/{epochs}'):            
            # Sample once from each buffer
            data = worker.get_sample()
            agent.learn_on_batch(data)

        total_steps += epoch_steps

        if hasattr(agent, "post_epoch_process"):
            agent.post_epoch_process()

        history_It, history_Rain, history_st, history_ET_o, history_ETmax, history_Kc, history_rho = worker.eval(eval_episode_num=training_params['evaluate_episode_num'])

        _log_metrics(logger, epoch, total_steps, time.time() - start_time)

        # Save the model periodically after epochs
        if (epoch % save_interval == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
            plot_simulation_data(history_It, history_Rain, history_st, history_ET_o, history_ETmax, history_Kc, history_rho, env.S_STAR, env.SFC, env.SW, model_directory, epoch, model_name)
            plot_results(all_rewards, 'rewards', model_directory, epoch)
            plot_results(all_violations, 'violations', model_directory, epoch)
            agent.save_model()

    # Optionally, save final model
    agent.save_model()
    
    # Return performance metrics for optimization
    total_violations = sum(violations)
    avg_violations = total_violations / len(violations) if violations else float('inf')

    return {
        'total_violations': total_violations,
        'avg_violations_per_episode': avg_violations,
        'final_violations': violations[-10:] if len(violations) >= 10 else violations,  # Last 10 episodes
        'total_episodes': len(violations)
    }