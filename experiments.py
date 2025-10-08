import os
import pandas as pd
import argparse
from water_environment import WaterEnvironment
from training import train_agent
from setup_parameters import setup_experiment_parameters  # Import the wrapper function


def setup_directories_and_data():
    """Setup directories and load weather data."""
    folder_path_initial = '/home/egomez/irrigation_project/'
    folder_path_output = '/scratch/egomez/irrigation_project_output/'
    data_file = os.path.join(folder_path_initial, 'daily_weather_data.csv')
    model_directory = os.path.join(folder_path_output, 'models')
    
    os.makedirs(model_directory, exist_ok=True)
    
    df = pd.read_csv(data_file)
    
    return df, model_directory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run irrigation experiments with different settings.')
    parser.add_argument('--n_days_ahead', type=int, required=True, 
                        help='Number of days ahead for the prediction')
    parser.add_argument('--chance_const', type=float, required=True, 
                        help='Chance constraint factor, try: 0.75, 0.85, 0.95')
    parser.add_argument('--model_type', type=str, required=True, 
                        help='Choose: DDPG, SAC, DDPGLagrangian or SACLagrangian')
    # parser.add_argument('--KP', type=float, required=True, 
    #                     help='Proportional gain for the controller')
    # parser.add_argument('--KI', type=float, required=True, 
    #                     help='Integral gain for the controller')
    # parser.add_argument('--KD', type=float, required=True, 
    #                     help='Derivative gain for the controller')
    return parser.parse_args()


def run_experiment(args, model_directory, base_env_params, base_agent_params, base_training_params):
    """Run the irrigation experiment with the given parameters."""
    # Generate experiment identifier
    identifier = (
        f"exp94_{args.model_type}"
        f"_chance_{args.chance_const}_days{args.n_days_ahead}"
    )
    
    # Setup experiment directory
    exp_dir = os.path.join(model_directory, identifier)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Run training directly with the parameters from setup_parameters
    train_agent(base_env_params, base_agent_params, base_training_params, exp_dir, identifier)
    
    return identifier, exp_dir


def main():
    """Main function to run irrigation experiments."""
    # Setup data and directories
    df, model_directory = setup_directories_and_data()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup experiment parameters using the imported function
    base_env_params, base_agent_params, base_training_params = setup_experiment_parameters(args, df)
    
    # Run experiment
    identifier, exp_dir = run_experiment(args, model_directory, base_env_params, base_agent_params, base_training_params)
    
    print(f"Experiment completed: {identifier}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()