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
    python hpo.py --n-days-ahead 7 --model-type DDPGLagrangian --chance-const 0.95
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileStorage
import gc
import torch
import json
from datetime import datetime
import logging
import traceback
from pathlib import Path

# Import custom modules
try:
    from env.water_environment import WaterEnvironment
    from models.safe_rl.train import train_agent
    from models.safe_rl.config import config as setup_parameters
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Ensure water_environment.py, train.py, and config.py are importable")

# Configure matplotlib for headless execution
plt.switch_backend('Agg')
plt.style.use('default')

class IrrigationOptimizationStudy:
    """
    Comprehensive hyperparameter optimization study for irrigation systems.
    
    Supports flexible configuration parameters and uses pre-defined parameters
    from setup_parameters.py while optimizing only PID controller parameters.
    """
    
    def __init__(self, n_days_ahead=7, model_type='DDPGLagrangian', chance_const=0.95,
                 base_path="/scratch/egomez/irrigation_hpo",
                 n_workers=1, worker_id=0):
        self.base_path = Path(base_path)
        self.n_workers = n_workers
        self.worker_id = worker_id

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
        self.data_file = '/home/egomez/irrigation_project/env/daily_weather_data.csv'
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Weather data file not found: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        
        # Study configuration
        self.study_config = {
            'max_total_trials': 500,
            'checkpoint_frequency': 5,
            'convergence_threshold': 0.05,
            'memory_cleanup_frequency': 3,
        }
        
        # Initialize optimization space - all 9 PID parameters for 3 constraints
        self.optimization_space = {
            # PID parameters for s_star constraint
            'KP_s_star': {'type': 'float', 'low': 0.01, 'high': 80, 'log': True},
            'KI_s_star': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KD_s_star': {'type': 'float', 'low': 0.001, 'high': 30, 'log': True},
            # PID parameters for sfc constraint  
            'KP_sfc': {'type': 'float', 'low': 0.01, 'high': 80, 'log': True},
            'KI_sfc': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KD_sfc': {'type': 'float', 'low': 0.001, 'high': 30, 'log': True},
            # PID parameters for sw constraint
            'KP_sw': {'type': 'float', 'low': 0.01, 'high': 80, 'log': True},
            'KI_sw': {'type': 'float', 'low': 0.001, 'high': 50, 'log': True},
            'KD_sw': {'type': 'float', 'low': 0.001, 'high': 30, 'log': True},
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
        """Configure logging for the study (per-worker log file)."""
        log_file = self.output_dir / 'logs' / (
            f'optimization_w{self.worker_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
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
            'num_epochs': 100,
            'evaluate_episode_num': 1,
            'plot_save_frequency': 10000,
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
        Optuna objective function for PID hyperparameter optimisation.

        Uses a smooth quadratic-penalty formulation:

            cost = -eval_reward  +  w · max(0, viol_rate - acceptable_rate)²

        - When feasible (violation rate ≤ acceptable), the penalty term is
          zero and Optuna simply minimises negative reward (= maximises
          water savings).
        - When infeasible, the quadratic penalty grows smoothly with the
          excess violation rate, giving the TPE sampler a useful gradient
          signal near the feasibility boundary.
        - ``w`` (PENALTY_WEIGHT) is set so that a 10 % excess violation
          adds a penalty roughly equal to the reward range, ensuring
          constraint satisfaction dominates but the landscape stays smooth.

        Returns:
            float: Scalar cost to minimise (lower is better).
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
            
            # Train agent (creates its own env internally)
            # Vary seed per trial so parallel workers explore different trajectories
            training_params = self.base_training_params.copy()
            training_params['seed'] = 42 + trial.number

            results = train_agent(
                self.base_env_params, 
                agent_params, 
                training_params, 
                str(trial_dir), 
                trial_id
            )
            
            # ---- Smooth constraint-aware objective (quadratic penalty) ----
            #
            # Goal: find PID gains so the policy satisfies the chance constraint
            # at the configured threshold while maximising reward.
            #
            # 1. Compute violation RATE (not raw count) so the objective is
            #    scale-invariant across different n_days_ahead horizons.
            # 2. Compare against the acceptable rate (1 - chance_const).
            # 3. Apply a smooth quadratic penalty for any excess violation,
            #    keeping the landscape differentiable for the TPE surrogate.
            #
            PENALTY_WEIGHT = 1e5  # scales so 10% excess ≈ 1e3 penalty
            
            best_violations = results.get('best_eval_violations', float('inf'))
            best_reward = results.get('best_eval_reward', float('-inf'))
            mean_ep_len = float(np.mean(results.get('all_eval_ep_lens', [1.0])))

            # Violation rate: fraction of unsafe decision steps
            violation_rate = best_violations / max(mean_ep_len, 1.0)
            acceptable_rate = 1.0 - self.config['chance_const']

            # Smooth quadratic penalty (zero when feasible)
            excess = max(0.0, violation_rate - acceptable_rate)
            feasible = excess == 0.0
            if feasible:
                avg_cost = -float(best_reward)
                penalty = 0.0
            else:
                penalty = 1.0 + PENALTY_WEIGHT * excess ** 2
                avg_cost = penalty

            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'optimized_parameters': sampled_params,
                'all_agent_parameters': {k: v for k, v in agent_params.items() if k not in ['env', 'weather_data']},
                'avg_cost': avg_cost,
                'best_eval_violations': best_violations,
                'best_eval_reward': best_reward,
                'violation_rate': violation_rate,
                'acceptable_rate': acceptable_rate,
                'excess_violation': excess,
                'penalty': penalty,
                'feasible': feasible,
                'mean_eval_ep_len': mean_ep_len,
                'duration_minutes': (datetime.now() - trial_start_time).total_seconds() / 60,
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config
            }
            
            with open(trial_dir / 'trial_results.json', 'w') as f:
                json.dump(trial_results, f, indent=2, default=lambda o: float(o))
            
            self.logger.info(
                f"Trial {trial.number} completed: cost={avg_cost:.4f}  "
                f"viol_rate={violation_rate:.3f}  acceptable={acceptable_rate:.3f}  "
                f"penalty={penalty:.2f}  "
                f"{'FEASIBLE' if feasible else 'INFEASIBLE'}  "
                f"eval_reward={best_reward:.2f}"
            )
            
            # Cleanup
            del results
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
                json.dump(error_info, f, indent=2, default=lambda o: float(o))
            
            self.cleanup_memory()
            return 1000.0  # High cost for failed trials
    
    def run_optimization(self, n_trials=None, force_new=False):
        """
        Run the Bayesian optimization study.

        Uses Optuna JournalFileStorage so that multiple workers sharing the
        same configuration directory can optimise concurrently — each worker
        simply calls study.optimize() and the journal file handles
        synchronisation.

        Args:
            n_trials: Total target number of trials across all workers.
            force_new: If True, delete any existing journal and start fresh.

        Returns:
            optuna.Study: The (potentially shared) study object.
        """
        if n_trials is None:
            n_trials = self.study_config['max_total_trials']

        # ---- Storage: journal file supports concurrent multi-worker access ----
        journal_path = str(self.output_dir / 'checkpoints' / 'optuna_journal.log')

        if force_new and self.worker_id == 0:
            journal_file = Path(journal_path)
            if journal_file.exists():
                journal_file.unlink()
                self.logger.info("Deleted existing journal (--force-new-study)")

        storage = JournalStorage(JournalFileStorage(journal_path))

        # ---- Create / load study (load_if_exists=True is the key) ----
        study_name = f'irrigation_optimization_{self.config_name}'
        sampler = TPESampler(seed=42 + self.worker_id, n_startup_trials=10)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            sampler=sampler,
            load_if_exists=True,
        )

        # ---- Determine how many trials THIS worker should run ----
        already_done = len([
            t for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE,
                           optuna.trial.TrialState.RUNNING)
        ])
        remaining_global = max(0, n_trials - already_done)
        trials_this_worker = max(1, remaining_global // max(self.n_workers, 1))
        # Worker 0 picks up the remainder
        if self.worker_id == 0:
            trials_this_worker += remaining_global % max(self.n_workers, 1)

        if remaining_global == 0:
            self.logger.info(
                f"Worker {self.worker_id}: study already has {already_done} "
                f"trials — target {n_trials} reached, skipping."
            )
        else:
            self.logger.info(
                f"Worker {self.worker_id}: running {trials_this_worker} trials "
                f"(global target: {n_trials}, already done: {already_done})"
            )

            # ---- Convergence callback — calls study.stop() for this worker ----
            threshold = self.study_config['convergence_threshold']

            def _convergence_cb(study, trial):
                completed = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if len(completed) >= 20 and len(completed) % 5 == 0:
                    recent = [t.value for t in completed[-20:]]
                    mean_v = np.mean(recent)
                    cv = np.std(recent) / mean_v if mean_v > 0 else float('inf')
                    if cv <= threshold:
                        self.logger.info(
                            f"Worker {self.worker_id}: convergence "
                            f"(CV={cv:.4f} <= {threshold})"
                        )
                        study.stop()

            # ---- Run ----
            session_start = datetime.now()
            try:
                study.optimize(
                    self.objective,
                    n_trials=trials_this_worker,
                    callbacks=[_convergence_cb],
                    show_progress_bar=False,
                    gc_after_trial=True,
                )
            except KeyboardInterrupt:
                self.logger.info(
                    f"Worker {self.worker_id}: interrupted after "
                    f"{len(study.trials)} total trials"
                )
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id}: {e}")
                self.logger.error(traceback.format_exc())

            elapsed = (datetime.now() - session_start).total_seconds() / 60
            self.logger.info(
                f"Worker {self.worker_id}: finished in {elapsed:.1f} min "
                f"({len(study.trials)} total trials in study)"
            )

        # ---- Only worker 0 writes the final results / plots ----
        if self.worker_id == 0:
            self.generate_results(study)

        return study

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
            json.dump(summary, f, indent=2, default=lambda o: float(o))
        
        # Generate visualizations
        if len(completed_trials) > 1:
            self.create_visualizations(study, results_dir)
        
        # Generate parameter importance analysis
        if len(completed_trials) > 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                with open(results_dir / 'parameter_importance.json', 'w') as f:
                    json.dump(importance, f, indent=2, default=lambda o: float(o))
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
            json.dump(trial_data, f, indent=2, default=lambda o: float(o))
        
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
                        help='Total target number of trials (default: use config setting)')
    parser.add_argument('--base-path', type=str, default='/scratch/egomez/irrigation_hpo',
                        help='Base output path')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Number of parallel workers for this configuration')
    parser.add_argument('--worker-id', type=int, default=0,
                        help='Worker index (0-based) — each worker gets a distinct sampler seed')
    parser.add_argument('--force-new-study', action='store_true', default=False,
                        help='Delete existing journal and start a fresh study')
    args = parser.parse_args()
    
    print(f"Starting irrigation PID optimization study")
    print(f"  config    : {args.model_type}  days={args.n_days_ahead}  chance={args.chance_const}")
    print(f"  worker    : {args.worker_id}/{args.n_workers}")
    print(f"  base_path : {args.base_path}")
    print(f"  start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        study_manager = IrrigationOptimizationStudy(
            n_days_ahead=args.n_days_ahead,
            model_type=args.model_type,
            chance_const=args.chance_const,
            base_path=args.base_path,
            n_workers=args.n_workers,
            worker_id=args.worker_id,
        )
        
        study = study_manager.run_optimization(
            n_trials=args.trials,
            force_new=args.force_new_study,
        )
        
        # Print final summary (from any worker — the study is shared)
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        print(f"\n{'='*60}")
        print(f"WORKER {args.worker_id} FINISHED — {study_manager.config_name}")
        print(f"{'='*60}")
        print(f"Total trials in study: {len(study.trials)}")
        
        if completed_trials:
            best = study.best_trial
            feasible = best.value < 1e6
            print(f"Best cost: {study.best_value:.4f}  (trial #{best.number})  "
                  f"{'FEASIBLE' if feasible else 'INFEASIBLE'}")
            for constraint in ('s_star', 'sfc', 'sw'):
                kp = study.best_params.get(f'KP_{constraint}', 0)
                ki = study.best_params.get(f'KI_{constraint}', 0)
                kd = study.best_params.get(f'KD_{constraint}', 0)
                print(f"  {constraint}: KP={kp:.6f}, KI={ki:.6f}, KD={kd:.6f}")
            n_feasible = sum(1 for t in completed_trials if t.value < 1e6)
            print(f"Feasible trials: {n_feasible}/{len(completed_trials)}")
        else:
            failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            print(f"No completed trials. Failed: {len(failed)}")
        
        print(f"Results: {study_manager.output_dir}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Study failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()