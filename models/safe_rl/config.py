import torch

def config(dataset, model, n_days_ahead, chance_const):
    """
    Setup parameters for different model types.

    Uses standard literature defaults for learning rates:
      - DDPG family:  actor=1e-4, critic=1e-3  (Lillicrap et al., 2016)
      - SAC family:   actor=3e-4, critic=3e-4   (Haarnoja et al., 2018)

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
        'polyak': 0.99,
        'model_type': model,
        'hidden_sizes': [128, 128],
        'chance_const': chance_const,
        'optimistic_regularization': 0.0,
        'temperature': 1.0,
    }

    # ── Default PID gains (used by Lagrangian models) ────────
    default_pid = {
        # s_star constraint
        'KP_s_star': 1.0,
        'KI_s_star': 0.1,
        'KD_s_star': 0.01,
        # sfc constraint
        'KP_sfc': 1.0,
        'KI_sfc': 0.1,
        'KD_sfc': 0.01,
        # sw constraint
        'KP_sw': 1.0,
        'KI_sw': 0.1,
        'KD_sw': 0.01,
    }

    # ── Model-specific parameters ────────────────────────────
    if model == 'DDPGLagrangian':
        base_agent_params = {
            **base_agent_defaults,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'cost_critic_lr': 1e-3,
            **default_pid,
        }

    elif model == 'SACLagrangian':
        base_agent_params = {
            **base_agent_defaults,
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'cost_critic_lr': 3e-3,
            **default_pid,
        }

    elif model == 'DDPG':
        base_agent_params = {
            **base_agent_defaults,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
        }

    elif model == 'SAC':
        base_agent_params = {
            **base_agent_defaults,
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
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