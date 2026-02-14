#!/usr/bin/env python3
"""
Hyperparameter Optimization for Safe Reinforcement Learning in Irrigation Systems

This script implements a comprehensive Bayesian hyperparameter optimization study
for a safe reinforcement learning irrigation system. Designed to run as a batch job
on computing clusters with automatic checkpointing and result management.

Features:
- Bayesian optimization with Optuna
- Configuration-specific output directories
- Automatic checkpointing and recovery
- Memory management for long runs
- Comprehensive result analysis
- Cluster-friendly execution

Usage:
    python irrigation_optimization_cluster.py --n-days-ahead 7 --model-type DDPGLagrangian --chance-const 0.95 --safe-buffer
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import gc
import torch
import pickle
import json
from datetime import datetime
import logging
import traceback
from pathlib import Path

# Import custom modules (assuming they exist in the project)
try:
    from data.water_environment import WaterEnvironment
    from safe_rl.train import train_agent
    from models.safe_rl.config import setup_parameters  # Import the setup function
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Ensure water_environment.py, training.py, and setup_parameters.py are in the Python path")

# Configure matplotlib for headless execution
plt.switch_backend('Agg')
plt.style.use('default')

class IrrigationOptimizationStudy:
    """
    Comprehensive hyperparameter optimization study for irrigation systems.
    
    Supports flexible configuration parameters and uses pre-defined parameters
    from setup_parameters.py while optimizing only PID controller parameters.
    """
    
    def __init__(self, n_days_ahead=7, model_type='DDPGLagrangian', chance_const=0.95, base_path="/scratch/egomez/irrigation_project_output"):
        self.base_path = Path(base_path)

        if n_days_ahead < 0 or n_days_ahead > 7:
            raise ValueError("n_days_ahead must be between 0 and 7")

        if chance_const < 0 or chance_const > 1.0:
            raise ValueError("chance_const must be between 0 and 1.0")

        if model_type not in ['DDPGLagrangian', 'SACLagrangian', 'SAC', 'DDPG']:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type == "SAC" or model_type == "DDPG":
            raise ValueError("You chose a regular RL algorithm, this study is not intended for this purpose.")

        self.config = {
            'n_days_ahead': n_days_ahead,
            'model_type': model_type,
            'chance_const': chance_const,
        }

        # Generate config name for logging and directory naming
        self.config_name = self._generate_config_name()

        # Create configuration-specific directory
        self.output_dir = self.base_path / self._get_config_dirname()
        self.setup_directories()
        self.setup_logging()
        
        # Load data
        self.data_file = '/home/egomez/irrigation_project/daily_weather_data.csv'
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Weather data file not found: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        
        # Study configuration
        self.study_config = {
            'max_total_trials': 75,
            'checkpoint_frequency': 5,
            'convergence_threshold': 0.05,
            'memory_cleanup_frequency': 3,
        }
        
        # Initialize optimization space - all 9 PID parameters for 3 constraints
        self.optimization_space = {
            # PID parameters for s_star constraint
            'KP_s_star': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KI_s_star': {'type': 'float', 'low': 0.0001, 'high': 10, 'log': True},
            'KD_s_star': {'type': 'float', 'low': 0.0001, 'high': 1, 'log': True},
            # PID parameters for sfc constraint  
            'KP_sfc': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KI_sfc': {'type': 'float', 'low': 0.0001, 'high': 10, 'log': True},
            'KD_sfc': {'type': 'float', 'low': 0.0001, 'high': 1, 'log': True},
            # PID parameters for sw constraint
            'KP_sw': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KI_sw': {'type': 'float', 'low': 0.0001, 'high': 10, 'log': True},
            'KD_sw': {'type': 'float', 'low': 0.0001, 'high': 1, 'log': True},
        }
        
        # Setup base parameters using setup_parameters.py
        self.setup_base_parameters()
        
        self.logger.info(f"Initialized optimization study for configuration: {self.config_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: {self.config}")
        self.logger.info(f"Optimizing 9 PID parameters for 3 constraints: {list(self.optimization_space.keys())}")
    
    def _generate_config_name(self):
        """Generate a descriptive name for the configuration."""
        config = self.config
        return f"{config['model_type']}_days{config['n_days_ahead']}_chance{int(config['chance_const']*100)}"
    
    def _get_config_dirname(self):
        """Generate directory name based on configuration parameters."""
        return f"PID_optimization_{self._generate_config_name()}"
    
    def setup_directories(self):
        """Create necessary directories for the study."""
        directories = [
            self.output_dir,
            self.output_dir / 'models',
            self.output_dir / 'trials',
            self.output_dir / 'checkpoints',
            self.output_dir / 'results',
            self.output_dir / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging for the study."""
        log_file = self.output_dir / 'logs' / f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_base_parameters(self):
        """Setup base parameters using the setup_parameters function."""
        
        # Get pre-defined parameters from setup_parameters.py
        base_env_params, base_agent_params, base_training_params = setup_parameters(
            dataset=self.df,
            model=self.config['model_type'],
            n_days_ahead=self.config['n_days_ahead'],
            chance_const=self.config['chance_const']
        )
        
        # Store the base parameters
        self.base_env_params = base_env_params
        self.base_agent_params = base_agent_params
        self.base_training_params = base_training_params
        
        # Device detection - fix the device assignment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Update training parameters for cluster execution
        self.base_training_params.update({
            'device': self.device,
            'num_epochs': 200,  # Reduced for optimization
            'evaluate_episode_num': 1,
            'plot_save_frequency': 1000,
            'sample_episode_num': 20,
            'episode_rerun_num': 5,
            'max_episode_steps': 1000,
            'batch_size': 512,
            'seed': 42,
            'threads': 1,
        })
        
        self.logger.info(f"Base agent parameters loaded from setup_parameters.py:")
        for key, value in self.base_agent_params.items():
            if key not in ['env', 'weather_data']:  # Skip large objects in logging
                self.logger.info(f"  {key}: {value}")
    
    def cleanup_memory(self):
        """Clean up GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def objective(self, trial):
        """
        Optuna objective function for Bayesian hyperparameter optimization.
        Optimizes all 9 PID parameters (3 constraints × 3 PID gains each) while using 
        pre-defined parameters for everything else.
        
        Args:
            trial: Optuna trial object for hyperparameter sampling
            
        Returns:
            float: Average cost per episode (to minimize)
        """
        trial_start_time = datetime.now()
        self.cleanup_memory()
        
        # Sample all 9 PID hyperparameters
        sampled_params = {}
        for param_name, param_config in self.optimization_space.items():
            if param_config['log']:
                sampled_params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=True
                )
            else:
                sampled_params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high']
                )
        
        # Create agent parameters by merging pre-defined params with optimized PID values
        agent_params = self.base_agent_params.copy()
        agent_params.update(sampled_params)
        
        # Create unique trial directory
        trial_id = f"trial_{trial.number:04d}_{datetime.now().strftime('%H%M%S')}"
        trial_dir = self.output_dir / 'trials' / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Starting trial {trial.number}")
            self.logger.info(f"Optimized PID parameters:")
            self.logger.info(f"  s_star: KP={sampled_params['KP_s_star']:.4f}, KI={sampled_params['KI_s_star']:.4f}, KD={sampled_params['KD_s_star']:.4f}")
            self.logger.info(f"  sfc:    KP={sampled_params['KP_sfc']:.4f}, KI={sampled_params['KI_sfc']:.4f}, KD={sampled_params['KD_sfc']:.4f}")
            self.logger.info(f"  sw:     KP={sampled_params['KP_sw']:.4f}, KI={sampled_params['KI_sw']:.4f}, KD={sampled_params['KD_sw']:.4f}")
            
            # Create environment
            env = WaterEnvironment(**self.base_env_params)
            agent_params.update({
                'env': env,
                'chkpt_dir': str(trial_dir),
            })
            
            # Train agent
            results = train_agent(
                self.base_env_params, 
                agent_params, 
                self.base_training_params, 
                str(trial_dir), 
                trial_id
            )
            
            # Extract performance metric - handle potential missing keys
            if 'avg_cost_per_episode' in results:
                avg_cost = results['avg_cost_per_episode']
            elif 'avg_cost' in results:
                avg_cost = results['avg_cost']
            else:
                # Fallback: calculate from total costs if available
                total_costs = results.get('total_costs', [])
                if isinstance(total_costs, list) and len(total_costs) > 0:
                    avg_cost = np.mean(total_costs)
                else:
                    self.logger.warning("No cost metric found in results, using default high cost")
                    avg_cost = 1000.0
            
            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'optimized_parameters': sampled_params,
                'all_agent_parameters': {k: v for k, v in agent_params.items() if k not in ['env', 'weather_data']},
                'avg_cost': avg_cost,
                'total_costs': results.get('total_costs', avg_cost),
                'duration_minutes': (datetime.now() - trial_start_time).total_seconds() / 60,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config
            }
            
            with open(trial_dir / 'trial_results.json', 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            self.logger.info(f"Trial {trial.number} completed: cost={avg_cost:.6f}")
            
            # Cleanup
            del env, results
            self.cleanup_memory()
            
            return avg_cost
            
        except Exception as e:
            trial_duration = (datetime.now() - trial_start_time).total_seconds() / 60
            self.logger.error(f"Trial {trial.number} failed after {trial_duration:.1f} minutes: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Save error information
            error_info = {
                'trial_number': trial.number,
                'optimized_parameters': sampled_params,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'duration_minutes': trial_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(trial_dir / 'trial_error.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            
            self.cleanup_memory()
            return 1000.0  # High cost for failed trials
    
    def _is_study_compatible(self, study):
        """
        Check if an existing study is compatible with the current optimization space.
        
        Args:
            study: Optuna study object
            
        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            # Check if we have any existing trials
            if len(study.trials) == 0:
                return True  # Empty study is always compatible
            
            # Get trials to check parameter compatibility
            all_trials = study.trials
            completed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
            
            # Use any available trial (completed or failed) to check parameter names
            reference_trial = None
            if completed_trials:
                reference_trial = completed_trials[0]
            elif failed_trials:
                reference_trial = failed_trials[0]
            
            if reference_trial is None:
                return True  # No trials with parameters yet
            
            # Check if parameter names match exactly
            existing_params = set(reference_trial.params.keys())
            expected_params = set(self.optimization_space.keys())
            
            if existing_params != expected_params:
                self.logger.warning(f"Parameter name mismatch. Expected: {expected_params}, Found: {existing_params}")
                return False
            
            # Check if parameter ranges and log settings match by trying to recreate the study
            try:
                # Try to create a dummy trial with current parameter space
                for param_name, param_config in self.optimization_space.items():
                    # This will fail if the existing study has different log settings or ranges
                    if param_config['log']:
                        study._storage.get_trial_param(reference_trial._trial_id, param_name)
                    else:
                        study._storage.get_trial_param(reference_trial._trial_id, param_name)
                
                # Additional check: try to access best_params which will fail if distributions are incompatible
                if completed_trials:
                    _ = study.best_params
                    
            except Exception as e:
                self.logger.warning(f"Parameter distribution compatibility check failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Compatibility check failed: {e}")
            return False
    
    def _delete_incompatible_studies(self):
        """Delete all study-related files to ensure clean start."""
        try:
            study_file = self.output_dir / 'checkpoints' / 'study_checkpoint.pkl'
            if study_file.exists():
                study_file.unlink()
                self.logger.info("Deleted incompatible study checkpoint")
            
            # Also clean up any backup files older than current session
            checkpoints_dir = self.output_dir / 'checkpoints'
            if checkpoints_dir.exists():
                for backup_file in checkpoints_dir.glob('study_backup_*.pkl'):
                    try:
                        backup_file.unlink()
                        self.logger.info(f"Deleted old backup: {backup_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not delete backup {backup_file.name}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Error during study cleanup: {e}")
    
    def _save_study_fingerprint(self):
        """Save a fingerprint of the current optimization configuration for future compatibility checks."""
        fingerprint = {
            'optimization_space': self.optimization_space,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'study_version': '1.0'  # Version for future compatibility
        }
        
        fingerprint_file = self.output_dir / 'checkpoints' / 'study_fingerprint.json'
        with open(fingerprint_file, 'w') as f:
            json.dump(fingerprint, f, indent=2)
    
    def _check_study_fingerprint(self):
        """Check if the saved study fingerprint matches current configuration."""
        fingerprint_file = self.output_dir / 'checkpoints' / 'study_fingerprint.json'
        
        if not fingerprint_file.exists():
            return False
        
        try:
            with open(fingerprint_file, 'r') as f:
                saved_fingerprint = json.load(f)
            
            # Check if optimization space matches exactly
            if saved_fingerprint.get('optimization_space') != self.optimization_space:
                self.logger.warning("Optimization space mismatch in fingerprint")
                return False
            
            # Check if configuration matches exactly
            if saved_fingerprint.get('config') != self.config:
                self.logger.warning("Configuration mismatch in fingerprint")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not read study fingerprint: {e}")
            return False

    def run_optimization(self, n_trials=None):
        """
        Run the Bayesian optimization study.
        
        Args:
            n_trials: Number of trials to run (uses study_config if None)
            
        Returns:
            optuna.Study: Completed optimization study
        """
        if n_trials is None:
            n_trials = self.study_config['max_total_trials']
        
        self.logger.info(f"Starting optimization with {n_trials} trials")
        self.logger.info(f"Configuration: {self.config}")
        
        # Study checkpoint file
        study_file = self.output_dir / 'checkpoints' / 'study_checkpoint.pkl'
        
        # Check fingerprint first for quick compatibility check
        fingerprint_matches = self._check_study_fingerprint()
        
        # Try to load existing study
        study = None
        start_trial = 0
        
        if fingerprint_matches:
            try:
                with open(study_file, 'rb') as f:
                    study = pickle.load(f)
                
                # Double-check compatibility with actual study
                if self._is_study_compatible(study):
                    self.logger.info(f"Loaded existing compatible study with {len(study.trials)} trials")
                    start_trial = len(study.trials)
                else:
                    self.logger.warning("Study failed detailed compatibility check despite matching fingerprint")
                    study = None
                    
            except FileNotFoundError:
                self.logger.info("No existing study found")
            except Exception as e:
                self.logger.warning(f"Could not load existing study: {e}")
                study = None
        else:
            self.logger.info("Study fingerprint doesn't match current configuration")
        
        # If no compatible study found, delete old files and create new study
        if study is None:
            self.logger.info("Creating new study - cleaning up old files")
            self._delete_incompatible_studies()
            
            study_name = f'irrigation_optimization_{self.config_name}_{datetime.now().strftime("%Y%m%d_%H%M")}'
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42, n_startup_trials=10),
                study_name=study_name
            )
            start_trial = 0
            self.logger.info("Created new optimization study")
            
            # Save fingerprint for future compatibility checks
            self._save_study_fingerprint()
        else:
            # Ensure the study is not empty before accessing trials
            if len(study.trials) > 0:
                self.logger.info(f"Loaded study with {len(study.trials)} trials")
            else:
                self.logger.info("Loaded study is empty, starting optimization")
        
        # Save configuration info
        config_info = {
            'study_name': study.study_name,
            'configuration': self.config,
            'config_name': self.config_name,
            'optimization_space': self.optimization_space,
            'study_config': self.study_config,
            'start_time': datetime.now().isoformat(),
            'target_trials': n_trials,
            'base_parameters': {
                'env_params': {k: v for k, v in self.base_env_params.items() if k != 'weather_data'},
                'agent_params': self.base_agent_params,
                'training_params': self.base_training_params
            }
        }
        
        with open(self.output_dir / 'checkpoints' / 'study_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        session_start = datetime.now()
        
        try:
            for i in range(n_trials):
                trial_number = start_trial + i + 1
                self.logger.info(f"Starting trial {trial_number} ({i+1}/{n_trials} in this session)")
                
                # Run single trial
                study.optimize(self.objective, n_trials=1, show_progress_bar=False)
                
                # Memory cleanup
                if (i + 1) % self.study_config['memory_cleanup_frequency'] == 0:
                    self.cleanup_memory()
                
                # Save checkpoint
                if (i + 1) % self.study_config['checkpoint_frequency'] == 0 or i == n_trials - 1:
                    with open(study_file, 'wb') as f:
                        pickle.dump(study, f)
                    
                    session_duration = (datetime.now() - session_start).total_seconds() / 60
                    self.logger.info(f"Checkpoint saved after {len(study.trials)} total trials")
                    self.logger.info(f"Session duration: {session_duration:.1f} minutes")
                    
                    if study.best_trial:
                        self.logger.info(f"Current best cost: {study.best_value:.6f} (trial #{study.best_trial.number})")
                
                # Check for convergence
                if len(study.trials) >= 20 and (i + 1) % 10 == 0:
                    if self.check_convergence(study):
                        self.logger.info("Study converged - stopping optimization")
                        break
        
        except KeyboardInterrupt:
            self.logger.info(f"Optimization interrupted by user after {len(study.trials)} trials")
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Save final study
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        # Generate comprehensive results
        self.generate_results(study)
        
        session_duration = (datetime.now() - session_start).total_seconds() / 60
        self.logger.info(f"Optimization session completed in {session_duration:.1f} minutes")
        
        return study
    
    def _clear_incompatible_study_files(self):
        """Clear study checkpoint files when parameter space changes."""
        self.logger.info("Force clearing all study files")
        self._delete_incompatible_studies()
        
        # Also delete fingerprint
        fingerprint_file = self.output_dir / 'checkpoints' / 'study_fingerprint.json'
        if fingerprint_file.exists():
            fingerprint_file.unlink()
            self.logger.info("Deleted study fingerprint")

    def check_convergence(self, study):
        """Check if the study has converged based on recent trials."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) < 20:
            return False
        
        recent_costs = [t.value for t in completed_trials[-20:]]
        cv = np.std(recent_costs) / np.mean(recent_costs) if np.mean(recent_costs) > 0 else float('inf')
        
        converged = cv <= self.study_config['convergence_threshold']
        self.logger.info(f"Convergence check: CV={cv:.4f}, threshold={self.study_config['convergence_threshold']}, converged={converged}")
        
        return converged
    
    def generate_results(self, study):
        """Generate comprehensive results and visualizations."""
        self.logger.info("Generating comprehensive results")
        
        results_dir = self.output_dir / 'results'
        
        # Save study results
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        # Get base parameters for reference
        base_env_params, base_agent_params, base_training_params = setup_parameters(
            dataset=self.df,
            model=self.config['model_type'],
            n_days_ahead=self.config['n_days_ahead'],
            chance_const=self.config['chance_const']
        )
        
        # Summary statistics
        summary = {
            'study_name': study.study_name,
            'configuration': self.config,
            'config_name': self.config_name,
            'optimization_space': self.optimization_space,
            'base_parameters': {
                'agent_params': {k: v for k, v in base_agent_params.items() if k not in ['env', 'weather_data']},
                'training_params': base_training_params
            },
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'failed_trials': len(failed_trials),
            'success_rate': len(completed_trials) / len(study.trials) * 100 if study.trials else 0,
            'completion_time': datetime.now().isoformat(),
            'best_value': study.best_value if len(completed_trials) > 0 else None,
            'best_trial_number': study.best_trial.number if len(completed_trials) > 0 else None,
            'best_parameters': study.best_params if len(completed_trials) > 0 else None
        }
        
        if completed_trials:
            costs = [t.value for t in completed_trials]
            summary.update({
                'cost_statistics': {
                    'mean': float(np.mean(costs)),
                    'std': float(np.std(costs)),
                    'min': float(np.min(costs)),
                    'max': float(np.max(costs)),
                    'median': float(np.median(costs)),
                    'improvement': float(np.max(costs) - np.min(costs)),
                    'improvement_percent': float((np.max(costs) - np.min(costs)) / np.max(costs) * 100)
                }
            })
            
            # Add comparison with base parameters
            if len(completed_trials) > 0:
                # Extract base PID parameters for all 3 constraints
                base_pid_params = {
                    's_star': {
                        'KP': base_agent_params.get('KP_s_star', 'N/A'),
                        'KI': base_agent_params.get('KI_s_star', 'N/A'), 
                        'KD': base_agent_params.get('KD_s_star', 'N/A')
                    },
                    'sfc': {
                        'KP': base_agent_params.get('KP_sfc', 'N/A'),
                        'KI': base_agent_params.get('KI_sfc', 'N/A'),
                        'KD': base_agent_params.get('KD_sfc', 'N/A')
                    },
                    'sw': {
                        'KP': base_agent_params.get('KP_sw', 'N/A'),
                        'KI': base_agent_params.get('KI_sw', 'N/A'),
                        'KD': base_agent_params.get('KD_sw', 'N/A')
                    }
                }
                
                # Organize optimized parameters by constraint
                optimized_pid_params = {
                    's_star': {
                        'KP': study.best_params.get('KP_s_star'),
                        'KI': study.best_params.get('KI_s_star'),
                        'KD': study.best_params.get('KD_s_star')
                    },
                    'sfc': {
                        'KP': study.best_params.get('KP_sfc'),
                        'KI': study.best_params.get('KI_sfc'),
                        'KD': study.best_params.get('KD_sfc')
                    },
                    'sw': {
                        'KP': study.best_params.get('KP_sw'),
                        'KI': study.best_params.get('KI_sw'),
                        'KD': study.best_params.get('KD_sw')
                    }
                }
                
                summary['parameter_comparison'] = {
                    'base_pid_parameters': base_pid_params,
                    'optimized_pid_parameters': optimized_pid_params,
                    'optimization_method': 'Bayesian Optimization with TPE Sampler',
                    'total_parameters_optimized': 9,
                    'constraints_optimized': ['s_star', 'sfc', 'sw']
                }
        
        # Save summary
        with open(results_dir / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualizations
        if len(completed_trials) > 1:
            self.create_visualizations(study, results_dir)
        
        # Generate parameter importance analysis
        if len(completed_trials) > 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                with open(results_dir / 'parameter_importance.json', 'w') as f:
                    json.dump(importance, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Could not generate parameter importance: {e}")
        
        # Save detailed trial data
        trial_data = []
        for trial in study.trials:
            trial_info = {
                'number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'params': trial.params,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            trial_data.append(trial_info)
        
        with open(results_dir / 'trial_data.json', 'w') as f:
            json.dump(trial_data, f, indent=2)
        
        self.logger.info(f"Results saved to {results_dir}")
    
    def create_visualizations(self, study, results_dir):
        """Create comprehensive visualization plots."""
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Check if we have enough data for visualization
        if len(completed_trials) < 2:
            self.logger.warning(f"Not enough completed trials ({len(completed_trials)}) for meaningful visualizations")
            return
            
        costs = [t.value for t in completed_trials]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Optimization Results - {self.config_name} Configuration\n{len(completed_trials)} Completed Trials', fontsize=16)
        
        # 1. Optimization progress
        axes[0, 0].plot(range(1, len(costs) + 1), costs, 'b-', alpha=0.7, linewidth=1, label='Trial costs')
        axes[0, 0].plot(range(1, len(costs) + 1), np.minimum.accumulate(costs), 'r-', linewidth=2, label='Best so far')
        axes[0, 0].set_title('Cost Minimization Progress')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Average Cost per Episode')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Cost distribution
        axes[0, 1].hist(costs, bins=min(20, len(costs)), alpha=0.7, color='skyblue', edgecolor='black')
        if len(completed_trials) > 0:
            best_value = study.best_value
            axes[0, 1].axvline(best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {best_value:.4f}')
        axes[0, 1].axvline(np.mean(costs), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(costs):.4f}')
        axes[0, 1].set_title('Cost Distribution')
        axes[0, 1].set_xlabel('Average Cost per Episode')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Convergence analysis
        if len(costs) > 10:
            window_size = min(10, len(costs) // 3)
            rolling_best = pd.Series(np.minimum.accumulate(costs)).rolling(window_size).mean()
            
            axes[1, 0].plot(range(1, len(costs) + 1), costs, 'b-', alpha=0.3, label='Individual trials')
            axes[1, 0].plot(range(1, len(costs) + 1), np.minimum.accumulate(costs), 'r-', linewidth=2, label='Best so far')
            axes[1, 0].plot(range(1, len(costs) + 1), rolling_best, 'g-', linewidth=2, label=f'Rolling mean (window={window_size})')
            
            # Convergence status
            recent_costs = costs[-min(20, len(costs) // 2):]
            if len(recent_costs) > 5:
                cv = np.std(recent_costs) / np.mean(recent_costs)
                convergence_status = "Converged" if cv < self.study_config['convergence_threshold'] else "Still improving"
                axes[1, 0].text(0.02, 0.98, f'Status: {convergence_status}\nCV: {cv:.3f}', 
                               transform=axes[1, 0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1, 0].set_title('Convergence Analysis')
        axes[1, 0].set_xlabel('Trial Number')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Parameter importance (if available)
        try:
            if len(completed_trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())
                importances = list(importance.values())
                
                # Sort by importance
                sorted_items = sorted(zip(params, importances), key=lambda x: x[1], reverse=True)
                params, importances = zip(*sorted_items)
                
                bars = axes[1, 1].barh(params, importances)
                axes[1, 1].set_title('Parameter Importance')
                axes[1, 1].set_xlabel('Importance Score')
                
                # Color bars by importance
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
            else:
                axes[1, 1].text(0.5, 0.5, 'Need >10 trials\nfor importance analysis', 
                                ha='center', va='center', transform=axes[1, 1].transAxes)
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Importance analysis\nfailed: {str(e)[:30]}...', 
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(results_dir / 'optimization_results.pdf', bbox_inches='tight')
        plt.close()
        
        self.logger.info("Visualizations saved")


def main():
    """Main function for cluster execution."""
    parser = argparse.ArgumentParser(description='Irrigation System PID Parameter Optimization')
    parser.add_argument('--n-days-ahead', type=int, default=7,
                        help='Number of days ahead for weather prediction (default: 7)')
    parser.add_argument('--model-type', type=str, default='DDPGLagrangian',
                        choices=['DDPGLagrangian', 'SACLagrangian', 'SAC', 'DDPG'],
                        help='Model type to use (default: DDPGLagrangian)')
    parser.add_argument('--chance-const', type=float, default=0.95,
                        help='Chance constraint value (0.0-1.0, default: 0.95)')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials to run (default: use config setting)')
    parser.add_argument('--base-path', type=str, default='/scratch/egomez/irrigation_project_output',
                        help='Base output path')
    parser.add_argument('--force-new-study', action='store_true', default=False,
                        help='Force creation of new study (ignore existing checkpoints)')
    args = parser.parse_args()
    
    print(f"Starting irrigation PID optimization study")
    print(f"n_days_ahead: {args.n_days_ahead}")
    print(f"model_type: {args.model_type}")
    print(f"chance_const: {args.chance_const}")
    print(f"force_new_study: {args.force_new_study}")
    print(f"Base path: {args.base_path}")
    print(f"Optimization focus: 9 PID controller parameters (KP, KI, KD for s_star, sfc, sw constraints)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize study with individual parameters
        study_manager = IrrigationOptimizationStudy(
            n_days_ahead=args.n_days_ahead,
            model_type=args.model_type,
            chance_const=args.chance_const,
            base_path=args.base_path
        )
        
        # Clear existing study if explicitly requested
        if args.force_new_study:
            print("Forcing new study - clearing all previous data")
            study_manager._clear_incompatible_study_files()
        else:
            print("Will try to resume from existing compatible study")
        
        # Run optimization (will automatically resume if compatible checkpoint exists)
        study = study_manager.run_optimization(n_trials=args.trials)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"PID OPTIMIZATION COMPLETED")
        print(f"{'='*60}")
        print(f"Configuration: {study_manager.config_name}")
        print(f"Total trials: {len(study.trials)}")
        
        # Check if we have any completed trials before accessing best_trial
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials) > 0:
            print(f"Best cost: {study.best_value:.6f}")
            print(f"Best trial: #{study.best_trial.number}")
            print(f"Optimized PID parameters:")
            
            # Organize output by constraint for clarity
            constraints = ['s_star', 'sfc', 'sw']
            for constraint in constraints:
                kp_key = f'KP_{constraint}'
                ki_key = f'KI_{constraint}'
                kd_key = f'KD_{constraint}'
                print(f"  {constraint}: KP={study.best_params.get(kp_key, 'N/A'):.6f}, "
                      f"KI={study.best_params.get(ki_key, 'N/A'):.6f}, "
                      f"KD={study.best_params.get(kd_key, 'N/A'):.6f}")
        else:
            print("No trials completed successfully.")
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            print(f"Failed trials: {len(failed_trials)}")
        
        print(f"Results saved to: {study_manager.output_dir}")
        print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Study failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()