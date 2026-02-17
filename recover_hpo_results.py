#!/usr/bin/env python3
"""
Recover HPO results from existing Optuna journal files.

The previous tuning job was killed by wall-time before worker-0 of most
configs could call generate_results().  The journals are intact, so this
script loads each one and writes the summary JSON + plots.

Run on the login node (no SLURM needed):
    python recover_hpo_results.py
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from pathlib import Path
from datetime import datetime

BASE_PATH = Path("/scratch/egomez/irrigation_hpo")

# Discover all config directories
config_dirs = sorted(BASE_PATH.glob("PID_optimization_*"))
if not config_dirs:
    print("No PID_optimization_* directories found.")
    sys.exit(1)

for config_dir in config_dirs:
    journal_path = config_dir / "checkpoints" / "optuna_journal.log"
    results_dir = config_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_name = config_dir.name.replace("PID_optimization_", "")

    if not journal_path.exists():
        print(f"  SKIP {config_name}: no journal file")
        continue

    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"{'='*60}")

    # Load study from journal
    storage = JournalStorage(JournalFileStorage(str(journal_path)))
    study_name = f"irrigation_optimization_{config_name}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        # Try listing available studies
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        if summaries:
            study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
        else:
            print(f"  SKIP: no study found in journal")
            continue

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"  Total trials : {len(study.trials)}")
    print(f"  Completed    : {len(completed)}")
    print(f"  Failed       : {len(failed)}")

    if not completed:
        print(f"  SKIP: no completed trials")
        continue

    # ── Best trial summary ────────────────────────────────────
    best = study.best_trial
    feasible = best.value < 1e6
    print(f"  Best cost    : {study.best_value:.4f}  (trial #{best.number})  "
          f"{'FEASIBLE' if feasible else 'INFEASIBLE'}")

    for constraint in ('s_star', 'sfc', 'sw'):
        kp = study.best_params.get(f'KP_{constraint}', 0)
        ki = study.best_params.get(f'KI_{constraint}', 0)
        kd = study.best_params.get(f'KD_{constraint}', 0)
        print(f"    {constraint}: KP={kp:.6f}, KI={ki:.6f}, KD={kd:.6f}")

    for lr_key in ('actor_lr', 'critic_lr', 'cost_critic_lr'):
        val = study.best_params.get(lr_key)
        if val is not None:
            print(f"    {lr_key}={val:.2e}")

    n_feasible = sum(1 for t in completed if t.value < 1e6)
    print(f"  Feasible     : {n_feasible}/{len(completed)}")

    # ── Save summary JSON ─────────────────────────────────────
    costs = [t.value for t in completed]
    summary = {
        'study_name': study.study_name,
        'config_name': config_name,
        'total_trials': len(study.trials),
        'completed_trials': len(completed),
        'failed_trials': len(failed),
        'best_value': study.best_value,
        'best_trial_number': best.number,
        'best_parameters': study.best_params,
        'feasible': feasible,
        'n_feasible': n_feasible,
        'cost_statistics': {
            'mean': float(np.mean(costs)),
            'std': float(np.std(costs)),
            'min': float(np.min(costs)),
            'max': float(np.max(costs)),
            'median': float(np.median(costs)),
        },
        'recovered_at': datetime.now().isoformat(),
    }

    with open(results_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o))

    # ── Save trial data ───────────────────────────────────────
    trial_data = []
    for trial in study.trials:
        trial_data.append({
            'number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'params': trial.params,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': trial.duration.total_seconds() if trial.duration else None,
        })

    with open(results_dir / 'trial_data.json', 'w') as f:
        json.dump(trial_data, f, indent=2, default=lambda o: float(o))

    # ── Parameter importance ──────────────────────────────────
    if len(completed) > 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            with open(results_dir / 'parameter_importance.json', 'w') as f:
                json.dump(importance, f, indent=2, default=lambda o: float(o))
        except Exception as e:
            print(f"  Warning: importance analysis failed: {e}")

    # ── Visualisations ────────────────────────────────────────
    if len(completed) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'HPO Results — {config_name}\n{len(completed)} Completed Trials',
                     fontsize=16)

        # 1. Progress
        axes[0, 0].plot(range(1, len(costs)+1), costs, 'b-', alpha=0.5, lw=1, label='Trial cost')
        axes[0, 0].plot(range(1, len(costs)+1), np.minimum.accumulate(costs),
                        'r-', lw=2, label='Best so far')
        axes[0, 0].set(title='Cost Minimisation Progress', xlabel='Trial', ylabel='Cost')
        axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        # 2. Distribution
        axes[0, 1].hist(costs, bins=min(30, len(costs)), alpha=0.7,
                        color='skyblue', edgecolor='black')
        axes[0, 1].axvline(study.best_value, color='red', ls='--', lw=2,
                           label=f'Best: {study.best_value:.4f}')
        axes[0, 1].axvline(np.mean(costs), color='green', ls='--', lw=2,
                           label=f'Mean: {np.mean(costs):.4f}')
        axes[0, 1].set(title='Cost Distribution', xlabel='Cost', ylabel='Count')
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        # 3. Convergence
        if len(costs) > 10:
            ws = min(10, len(costs)//3)
            rolling = pd.Series(np.minimum.accumulate(costs)).rolling(ws).mean()
            axes[1, 0].plot(range(1, len(costs)+1), costs, 'b-', alpha=0.3)
            axes[1, 0].plot(range(1, len(costs)+1), np.minimum.accumulate(costs),
                            'r-', lw=2, label='Best so far')
            axes[1, 0].plot(range(1, len(costs)+1), rolling, 'g-', lw=2,
                            label=f'Rolling mean (w={ws})')
            axes[1, 0].legend()
        axes[1, 0].set(title='Convergence', xlabel='Trial', ylabel='Cost')
        axes[1, 0].grid(alpha=0.3)

        # 4. Importance
        try:
            if len(completed) > 10:
                imp = optuna.importance.get_param_importances(study)
                params, vals = zip(*sorted(imp.items(), key=lambda x: x[1], reverse=True))
                bars = axes[1, 1].barh(params, vals)
                colours = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, c in zip(bars, colours):
                    bar.set_color(c)
                axes[1, 1].set(title='Parameter Importance', xlabel='Importance')
            else:
                axes[1, 1].text(0.5, 0.5, 'Need >10 trials', ha='center', va='center',
                                transform=axes[1, 1].transAxes)
        except Exception:
            axes[1, 1].text(0.5, 0.5, 'Importance N/A', ha='center', va='center',
                            transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Parameter Importance')

        plt.tight_layout()
        plt.savefig(results_dir / 'optimization_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(results_dir / 'optimization_results.pdf', bbox_inches='tight')
        plt.close()

    print(f"  Results saved to {results_dir}")

print(f"\nRecovery complete.")
