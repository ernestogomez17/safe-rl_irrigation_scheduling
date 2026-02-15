# Safe Deep Reinforcement Learning for Irrigation Scheduling

This repository contains the code for training and evaluating **safe deep reinforcement learning** agents for agricultural irrigation scheduling under soil-moisture constraints. A Monte-Carlo Stochastic Model Predictive Control (MC-SMPC) baseline is included for benchmarking.

## Overview

Irrigation scheduling is formulated as a constrained Markov Decision Process (CMDP). An RL agent observes weather forecasts and soil conditions, then decides daily irrigation amounts. Three soil-moisture constraints are enforced:

| Constraint | Type | Description |
|:-----------|:-----|:------------|
| **S\*** (stress onset) | Probabilistic | $P(s_t \geq S^*) \geq 1 - d$ |
| **SFC** (field capacity) | Probabilistic | $P(s_t \leq S_{FC}) \geq 1 - d$ |
| **SW** (wilting point) | Hard | $\mathbb{E}\left[\sum_t \gamma^t \mathbf{1}\{s_t < S_W\}\right] \leq 0$ |

The Lagrangian variants enforce these constraints using **PID-controlled Lagrange multipliers** (Stooke et al., 2020).

### Algorithms

| Algorithm | Description | Reference |
|:----------|:------------|:----------|
| **DDPG** | Deterministic policy gradient | Lillicrap et al., 2016 |
| **SAC** | Maximum-entropy actor-critic | Haarnoja et al., 2018 |
| **DDPG-Lagrangian** | DDPG + PID-Lagrangian constraints | Stooke et al., 2020 |
| **SAC-Lagrangian** | SAC + PID-Lagrangian constraints | Stooke et al., 2020 |
| **MC-SMPC** (baseline) | Monte-Carlo stochastic MPC | Roy et al., 2021 |

## Repository Structure

```
irrigation_scheduling_SafeRL/
├── main.py                          # Entry point for training RL agents
├── hop.py                           # Hyperparameter optimization (Optuna)
├── irrigation_job.sh                # SLURM script: full experiment grid
├── irrigation_tuning.sh             # SLURM script: hyperparameter search
│
├── env/                             # Environment
│   ├── params.py                    # Shared soil & crop parameters
│   ├── water_environment.py         # Gymnasium environment (soil-moisture dynamics)
│   └── daily_weather_data.csv       # Weather observations (2015–2018)
│
├── models/
│   ├── mc_irrigation_baseline.py    # MC-SMPC baseline controller
│   └── safe_rl/
│       ├── config.py                # Experiment configuration & hyperparameters
│       ├── train.py                 # Training loop (collect → update → evaluate)
│       ├── policy/
│       │   ├── ddpg.py              # DDPG agent
│       │   ├── ddpg_lag.py          # DDPG-Lagrangian agent
│       │   ├── sac.py               # SAC agent
│       │   ├── sac_lag.py           # SAC-Lagrangian agent
│       │   └── base_policy.py       # Abstract base class
│       ├── util/
│       │   ├── networks.py          # Actor, critic, ensemble Q networks
│       │   ├── pid_controller.py    # PID Lagrange multiplier controller
│       │   ├── logger.py            # Epoch-based metric logger
│       │   ├── torch_util.py        # Device & seed helpers
│       │   └── run_util.py          # Miscellaneous run utilities
│       └── worker/
│           ├── collector.py         # Environment interaction & data collection
│           ├── replay_buffer.py     # Experience replay buffer
│           └── segment_tree.py      # Segment tree for prioritised replay
│
└── plotting/
    └── generate_plots.py            # Training curve & simulation plots
```

## Environment

The `WaterEnvironment` is a custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates daily soil-moisture dynamics using the FAO-56 Penman-Monteith method for reference evapotranspiration. Key details:

- **Observation space** (9-dim): cyclical date encodings (sin/cos for month, day, week), soil moisture, normalised loss rate, normalised rainfall
- **Action space** (1-dim, continuous): irrigation amount in metres, scaled by the decision horizon
- **Episode**: one growing season (365 days), starting April 10
- **Soil physics**: sandy-loam soil with parameters defined in `env/params.py`
- **Crop**: grape (FAO-56 single-crop-coefficient $K_c$ schedule)

## Installation

```bash
# Clone the repository
git clone https://github.com/ernestogomez17/irrigation_scheduling_SafeRL.git
cd irrigation_scheduling_SafeRL

# Create virtual environment
python3.11 -m venv irrigation_env
source irrigation_env/bin/activate

# Install dependencies
pip install torch gymnasium numpy pandas matplotlib optuna
```

### Requirements

- Python 3.11
- PyTorch
- Gymnasium
- NumPy (< 2.0)
- Pandas
- Matplotlib
- Optuna (>= 3.1, for hyperparameter optimization)

## Usage

### Training a single agent

```bash
python main.py --n_days_ahead 7 --chance_const 0.95 --model_type SACLagrangian
```

| Argument | Type | Description |
|:---------|:-----|:------------|
| `--n_days_ahead` | int | Decision horizon: 1, 3, or 7 days |
| `--chance_const` | float | Chance constraint level (e.g. 0.75, 0.85, 0.95, 1.0) |
| `--model_type` | str | `DDPG`, `SAC`, `DDPGLagrangian`, or `SACLagrangian` |

Results are saved to `/scratch/egomez/irrigation_project_output/models/`.

### Running the MC-SMPC baseline

```bash
python models/mc_irrigation_baseline.py \
  --n-days-ahead 7 \
  --chance-pct 0.75 \
  --data env/daily_weather_data.csv \
  --out results/mc_baseline_days7_chance0.75.csv
```

### Full experiment grid (SLURM)

The job script runs **42 experiments** in parallel (24 Lagrangian + 6 standard RL + 12 MC-SMPC baselines):

```bash
sbatch irrigation_job.sh
```

### Hyperparameter optimization (SLURM)

Bayesian optimisation of PID controller gains using Optuna with parallel workers:

```bash
sbatch irrigation_tuning.sh
```

## Experiment Grid

The full experiment grid covers:

| Factor | Values |
|:-------|:-------|
| Decision horizon | 1, 3, 7 days |
| Chance constraint | 0.75, 0.85, 0.95, 1.0 |
| RL models (constrained) | SAC-Lagrangian, DDPG-Lagrangian |
| RL models (unconstrained) | SAC, DDPG |
| Baseline | MC-SMPC |

This produces **42 total runs**: 24 Lagrangian (2 models $\times$ 3 horizons $\times$ 4 constraints) + 6 standard RL (2 models $\times$ 3 horizons) + 12 baselines (3 horizons $\times$ 4 constraints).

## Key References

1. T. P. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning," *ICLR*, 2016.
2. T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," *ICML*, 2018.
3. A. Stooke, J. Achiam, and P. Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods," *ICML*, 2020.
4. A. Roy et al., "Short and Medium Range Irrigation Scheduling Using Stochastic Simulation-Optimization Framework With Farm-Scale Ecohydrological Model and Weather Forecasts," *Water Resources Research*, 2021.

## License

This project is provided for academic and research purposes.
