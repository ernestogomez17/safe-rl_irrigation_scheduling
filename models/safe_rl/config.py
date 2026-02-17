import torch

def config(dataset, model, n_days_ahead, chance_const):
    """
    Setup parameters for different model types.

    Learning rates and PID gains are fine-tuned per model family and
    decision frequency (n_days_ahead) via Bayesian HPO (Optuna, 2026-02).

    Args:
        dataset: Weather data DataFrame
        model: Model type string ('DDPGLagrangian', 'SACLagrangian', 'DDPG', 'SAC')
        n_days_ahead: Number of days ahead for prediction (1, 3, or 7)
        chance_const: Chance constraint value

    Returns:
        tuple: (base_env_params, base_agent_params, base_training_params)
    """

    base_env_params = {
        'weather_data': dataset,
        'n_days_ahead': n_days_ahead,
    }

    # ── Common defaults ──────────────────────────────────────
    base_agent_defaults = {
        'gamma': 0.99,
        'polyak': 0.98,
        'model_type': model,
        'hidden_sizes': [128, 128],
        'chance_const': chance_const,
        'optimistic_regularization': 0.3,
        'use_regularization_decay': True,
    }

    # ── Default PID gains (fallback for unseen n_days_ahead) ─
    default_pid = {
        'KP_s_star': 1.0, 'KI_s_star': 0.1, 'KD_s_star': 0.01,
        'KP_sfc':    1.0, 'KI_sfc':    0.1, 'KD_sfc':    0.01,
        'KP_sw':     1.0, 'KI_sw':     0.1, 'KD_sw':     0.01,
    }

    # ── Default learning rates (fallback for unseen n_days_ahead) ─
    default_ddpg_lr = {'actor_lr': 1e-4, 'critic_lr': 1e-3, 'cost_critic_lr': 1e-3}
    default_sac_lr  = {'actor_lr': 3e-4, 'critic_lr': 3e-4, 'cost_critic_lr': 3e-4}

    # ── Tuned params per model and decision frequency ────────
    # Fine-tuned via Bayesian HPO (Optuna TPE, 500 trials, 2026-02)

    ddpg_lag_params = {
        1: {
            'actor_lr': 1.95e-5, 'critic_lr': 6.15e-4, 'cost_critic_lr': 2.49e-3,
            'KP_s_star': 12.87, 'KI_s_star': 0.01, 'KD_s_star':  8.43,
            'KP_sfc':    55.72, 'KI_sfc':    0.03, 'KD_sfc':     0.19,
            'KP_sw':      2.02, 'KI_sw':     3.57, 'KD_sw':      0.02,
        },
        3: {
            'actor_lr': 3.41e-4, 'critic_lr': 1.45e-5, 'cost_critic_lr': 2.54e-3,
            'KP_s_star':  0.18, 'KI_s_star': 0.01, 'KD_s_star': 19.37,
            'KP_sfc':    77.43, 'KI_sfc':    0.16, 'KD_sfc':     0.02,
            'KP_sw':      0.52, 'KI_sw':     3.27, 'KD_sw':      0.00,
        },
        7: {
            'actor_lr': 2.92e-4, 'critic_lr': 2.29e-3, 'cost_critic_lr': 9.93e-3,
            'KP_s_star':  1.95, 'KI_s_star': 0.00, 'KD_s_star':  0.09,
            'KP_sfc':     9.63, 'KI_sfc':    0.02, 'KD_sfc':     0.20,
            'KP_sw':      3.19, 'KI_sw':     0.85, 'KD_sw':      0.04,
        },
    }

    sac_lag_params = {
        1: {
            'actor_lr': 7.09e-5, 'critic_lr': 1.55e-5, 'cost_critic_lr': 6.33e-3,
            'KP_s_star':  0.10, 'KI_s_star': 0.77, 'KD_s_star':  0.13,
            'KP_sfc':     0.06, 'KI_sfc':    0.18, 'KD_sfc':     0.63,
            'KP_sw':     21.01, 'KI_sw':     0.00, 'KD_sw':      0.22,
        },
        3: {
            'actor_lr': 1.17e-4, 'critic_lr': 3.37e-4, 'cost_critic_lr': 2.23e-3,
            'KP_s_star':  1.07, 'KI_s_star': 0.26, 'KD_s_star':  0.00,
            'KP_sfc':     0.01, 'KI_sfc':    0.39, 'KD_sfc':     1.11,
            'KP_sw':      0.95, 'KI_sw':     0.01, 'KD_sw':      0.02,
        },
        7: {
            'actor_lr': 4.39e-4, 'critic_lr': 3.42e-5, 'cost_critic_lr': 3.51e-3,
            'KP_s_star':  6.71, 'KI_s_star': 0.10, 'KD_s_star':  0.11,
            'KP_sfc':     2.00, 'KI_sfc':    0.16, 'KD_sfc':     0.16,
            'KP_sw':      0.04, 'KI_sw':     0.69, 'KD_sw':      3.84,
        },
    }

    # ── Tuned LRs for plain DDPG / SAC (from Lagrangian HPO) ─
    ddpg_lr = {
        1: {'actor_lr': 1.95e-5, 'critic_lr': 6.15e-4},
        3: {'actor_lr': 3.41e-4, 'critic_lr': 1.45e-5},
        7: {'actor_lr': 2.92e-4, 'critic_lr': 2.29e-3},
    }

    sac_lr = {
        1: {'actor_lr': 7.09e-5, 'critic_lr': 1.55e-5},
        3: {'actor_lr': 1.17e-4, 'critic_lr': 3.37e-4},
        7: {'actor_lr': 4.39e-4, 'critic_lr': 3.42e-5},
    }

    # ── Model-specific parameters ────────────────────────────
    if model == 'DDPGLagrangian':
        tuned = ddpg_lag_params.get(n_days_ahead, {**default_ddpg_lr, **default_pid})
        base_agent_params = {
            **base_agent_defaults,
            'temperature': 0.2,
            **tuned,
        }

    elif model == 'SACLagrangian':
        tuned = sac_lag_params.get(n_days_ahead, {**default_sac_lr, **default_pid})
        base_agent_params = {
            **base_agent_defaults,
            'temperature': 0.9,
            **tuned,
        }

    elif model == 'DDPG':
        lr = ddpg_lr.get(n_days_ahead, {'actor_lr': 1e-4, 'critic_lr': 1e-3})
        base_agent_params = {
            **base_agent_defaults,
            **lr,
        }

    elif model == 'SAC':
        lr = sac_lr.get(n_days_ahead, {'actor_lr': 3e-4, 'critic_lr': 3e-4})
        base_agent_params = {
            **base_agent_defaults,
            **lr,
        }

    else:
        raise ValueError(
            f"Unknown model type: {model}. "
            "Choose from 'DDPGLagrangian', 'SACLagrangian', 'DDPG', 'SAC'"
        )

    # ── Training parameters ──────────────────────────────────
    base_training_params = {
        'num_epochs': 250,
        'sample_episode_num': 20,
        'episode_rerun_num': 5,
        'evaluate_episode_num': 10,
        'plot_save_frequency': 100,
        'max_episode_steps': 1000,
        'batch_size': 512,
        'seed': 0,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'device_id': 0,
        'threads': 1,
    }

    return base_env_params, base_agent_params, base_training_params


def setup_experiment_parameters(args, df):
    """
    Wrapper function to maintain compatibility with experiments.py
    
    Args:
        args: Parsed command line arguments with attributes:
            - model_type: str
            - n_days_ahead: int  
            - chance_const: float
        df: Weather data DataFrame
        
    Returns:
        tuple: (base_env_params, base_agent_params, base_training_params)
    """
    # Call the main setup function with properly mapped arguments
    return config(
        dataset=df, 
        model=args.model_type, 
        n_days_ahead=args.n_days_ahead, 
        chance_const=args.chance_const
    )