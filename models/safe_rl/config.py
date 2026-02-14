import warnings

def config(dataset, model, n_days_ahead, chance_const):    
    """
    Setup optimized parameters for different model types and prediction horizons.
    
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
        'n_days_ahead': n_days_ahead
    }

    # Default base agent parameters that are common across models
    base_agent_defaults = {
        'gamma': 0.99,
        'model_type': model,
        'hidden_sizes': [256, 256],  # Changed from 'hidden_layers' to 'hidden_sizes'
        'chance_const': chance_const,
        'optimistic_regularization': 0.1,
        'temperature': 0.1,
    }

    # Model-specific and horizon-specific parameter configurations
    if model == 'DDPGLagrangian':
        if n_days_ahead == 1:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.97,
                'actor_lr': 4e-5,
                'critic_lr': 1e-5,
                'cost_critic_lr': 1e-5,
                # PID parameters for s_star constraint
                'KP_s_star': 0.001,
                'KI_s_star': 0.02,
                'KD_s_star': 0.3,
                # PID parameters for sfc constraint
                'KP_sfc': 3.5,
                'KI_sfc': 0.03,
                'KD_sfc': 0.09,
                # PID parameters for sw constraint
                'KP_sw': 0.9,
                'KI_sw': 0.001,
                'KD_sw': 0.003,
                'optimistic_regularization': 0.15,
                'temperature': 0.2,
            }
        
        elif n_days_ahead == 3:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.96,
                'actor_lr': 6e-6,
                'critic_lr': 7e-4,
                'cost_critic_lr': 6e-4,
                # PID parameters for s_star constraint
                'KP_s_star': 16,
                'KI_s_star': 0.005,
                'KD_s_star': 0.5,
                # PID parameters for sfc constraint
                'KP_sfc': 13,
                'KI_sfc': 0.02,
                'KD_sfc': 0.3,
                # PID parameters for sw constraint
                'KP_sw': 0.001,
                'KI_sw': 0.06,
                'KD_sw': 0.15,
                'optimistic_regularization': 0.18,
                'temperature': 0.15,
            }

        elif n_days_ahead == 7:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.99,
                'actor_lr': 7e-6,
                'critic_lr': 2e-4,
                'cost_critic_lr': 2e-4,
                # PID parameters for s_star constraint
                'KP_s_star': 30,
                'KI_s_star': 0.001,
                'KD_s_star': 0.6,
                # PID parameters for sfc constraint
                'KP_sfc': 30,
                'KI_sfc': 0.2,
                'KD_sfc': 0.0006,
                # PID parameters for sw constraint
                'KP_sw': 0.01,
                'KI_sw': 0.2,
                'KD_sw': 0.4,
                'optimistic_regularization': 0.28,
                'temperature': 0.3,
            }

        else:
            warnings.warn(f"Parameters haven't been optimized for n_days_ahead={n_days_ahead}. Using default parameters.")
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.95,
                'actor_lr': 1e-5,
                'critic_lr': 1e-4,
                'cost_critic_lr': 1e-3,
                # PID parameters for s_star constraint
                'KP_s_star': 0.5,
                'KI_s_star': 0.1,
                'KD_s_star': 0.002,
                # PID parameters for sfc constraint
                'KP_sfc': 0.5,
                'KI_sfc': 0.1,
                'KD_sfc': 0.002,
                # PID parameters for sw constraint
                'KP_sw': 0.5,
                'KI_sw': 0.1,
                'KD_sw': 0.002,
                'optimistic_regularization': 0.1,
                'temperature': 0.5,
            }

    elif model == 'SACLagrangian':
        if n_days_ahead == 1:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.96,
                'actor_lr': 1e-4,
                'critic_lr': 2e-4,
                'cost_critic_lr': 6e-4,
                # PID parameters for s_star constraint
                'KP_s_star': 24.3,
                'KI_s_star': 20.7,
                'KD_s_star': 2.16,
                # PID parameters for sfc constraint
                'KP_sfc': 0.3,
                'KI_sfc': 1.9,
                'KD_sfc': 0.2,
                # PID parameters for sw constraint
                'KP_sw': 52.5,
                'KI_sw': 48.8,
                'KD_sw': 6.4,
                'optimistic_regularization': 0.06,
                'temperature': 0.9,
            }
        
        elif n_days_ahead == 3:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.96,
                'actor_lr': 7e-5,
                'critic_lr': 2e-4,
                'cost_critic_lr': 3e-4,
                # PID parameters for s_star constraint
                'KP_s_star': 18.3,
                'KI_s_star': 15.2,
                'KD_s_star': 0.2,
                # PID parameters for sfc constraint
                'KP_sfc': 0.2,
                'KI_sfc': 0.8,
                'KD_sfc': 0.0006,
                # PID parameters for sw constraint
                'KP_sw': 43.6,
                'KI_sw': 36.5,
                'KD_sw': 1.12,
                'optimistic_regularization': 0.21,
                'temperature': 0.99,
            }

        elif n_days_ahead == 7:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.9,
                'actor_lr': 9e-5,
                'critic_lr': 2e-5,
                'cost_critic_lr': 7e-4,
                # PID parameters for s_star constraint
                'KP_s_star': 2.5,
                'KI_s_star': 1.8,
                'KD_s_star': 0.5,
                # PID parameters for sfc constraint
                'KP_sfc': 0.2,
                'KI_sfc': 0.2,
                'KD_sfc': 0.008,
                # PID parameters for sw constraint
                'KP_sw': 15,
                'KI_sw': 10.3,
                'KD_sw': 1.2,
                'optimistic_regularization': 0.29,
                'temperature': 0.96,
            }

        else:
            warnings.warn(f"Parameters haven't been optimized for n_days_ahead={n_days_ahead}. Using default parameters.")
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.99,
                'actor_lr': 3e-5,
                'critic_lr': 1e-4,
                'cost_critic_lr': 1e-3,
                # PID parameters for s_star constraint
                'KP_s_star': 0.3,
                'KI_s_star': 0.08,
                'KD_s_star': 0.001,
                # PID parameters for sfc constraint
                'KP_sfc': 0.3,
                'KI_sfc': 0.08,
                'KD_sfc': 0.001,
                # PID parameters for sw constraint
                'KP_sw': 0.3,
                'KI_sw': 0.08,
                'KD_sw': 0.001,
                'optimistic_regularization': 0.1,
                'temperature': 0.5,
            }

    elif model == 'DDPG':
        if n_days_ahead == 1:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.95,
                'actor_lr': 2e-5,
                'critic_lr': 2e-4,
            }
        
        elif n_days_ahead == 3:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.95,
                'actor_lr': 1e-5,
                'critic_lr': 3e-4,
            }

        elif n_days_ahead == 7:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.95,
                'actor_lr': 5e-6,
                'critic_lr': 4e-4,
            }

        else:
            warnings.warn(f"Parameters haven't been optimized for n_days_ahead={n_days_ahead}. Using default parameters.")
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.95,
                'actor_lr': 2e-5,
                'critic_lr': 2e-4,
            }

    elif model == 'SAC':
        if n_days_ahead == 1:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.99,
                'actor_lr': 3e-5,
                'critic_lr': 2e-4,
            }
        
        elif n_days_ahead == 3:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.99,
                'actor_lr': 2e-5,
                'critic_lr': 3e-4,
            }

        elif n_days_ahead == 7:
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.995,
                'actor_lr': 1e-5,
                'critic_lr': 4e-4,
            }

        else:
            warnings.warn(f"Parameters haven't been optimized for n_days_ahead={n_days_ahead}. Using default parameters.")
            base_agent_params = {
                **base_agent_defaults,
                'polyak': 0.99,
                'actor_lr': 3e-5,
                'critic_lr': 2e-4,
            }

    else:
        raise ValueError(f"Unknown model type: {model}. Choose from 'DDPGLagrangian', 'SACLagrangian', 'DDPG', 'SAC'")


    base_training_params = {
        'num_epochs': 500,
        'sample_episode_num': 20,
        'episode_rerun_num': 5,
        'evaluate_episode_num': 10,
        'plot_save_frequency': 100,
        'max_episode_steps': 1000,
        'batch_size': 512,
        'seed': 0,
        'device': "gpu",
        'device_id': 0,  # Added missing parameter
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
    return setup_parameters(
        dataset=df, 
        model=args.model_type, 
        n_days_ahead=args.n_days_ahead, 
        chance_const=args.chance_const
    )