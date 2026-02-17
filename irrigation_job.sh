#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --job-name=irrigation_project
#SBATCH --output=/scratch/egomez/irrigation_output/job_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ernesto.gomez@mail.utoronto.ca

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR="/home/egomez/irrigation_project"
PYTHON="${PROJECT_DIR}/irrigation_env/bin/python3"
OUTPUT_DIR="/scratch/egomez/irrigation_output"

mkdir -p "${OUTPUT_DIR}"

# ── Modules ──────────────────────────────────────────────────
module load StdEnv/2023
module load gcc/12.3

# ── Verify ───────────────────────────────────────────────────
cd "${PROJECT_DIR}"
echo "Python: ${PYTHON}"
${PYTHON} --version

# ── Generate commands (30 RL + 12 MC-MPC + 6 MPC = 48 total) ──────
generate_commands() {
  local N_DAYS_AHEAD_VALS=(1 3 7)
  local CHANCE_CONST_VALS=(0.75 0.85 0.95 1)
  local LAGRANGIAN_MODELS=("SACLagrangian" "DDPGLagrangian")
  local REGULAR_MODELS=("DDPG" "SAC")
  local MC_BASELINE="${PROJECT_DIR}/models/monte_carlo.py"
  local MPC_BASELINE="${PROJECT_DIR}/models/mpc.py"
  local DATA="${PROJECT_DIR}/env/daily_weather_data.csv"

  # Lagrangian models: all chance constraints
  for MODEL_TYPE in "${LAGRANGIAN_MODELS[@]}"; do
    for N_DAYS in "${N_DAYS_AHEAD_VALS[@]}"; do
      for CC in "${CHANCE_CONST_VALS[@]}"; do
        echo "${PYTHON} ${PROJECT_DIR}/main.py --n_days_ahead ${N_DAYS} --chance_const ${CC} --model_type ${MODEL_TYPE}"
      done
    done
  done

  # Regular models: chance_const=1 only
  for MODEL_TYPE in "${REGULAR_MODELS[@]}"; do
    for N_DAYS in "${N_DAYS_AHEAD_VALS[@]}"; do
      echo "${PYTHON} ${PROJECT_DIR}/main.py --n_days_ahead ${N_DAYS} --chance_const 1 --model_type ${MODEL_TYPE}"
    done
  done

  # MC-MPC baseline: all horizon × chance combinations
  for N_DAYS in "${N_DAYS_AHEAD_VALS[@]}"; do
    for CC in "${CHANCE_CONST_VALS[@]}"; do
      local MC_DIR="${OUTPUT_DIR}/mc_mpc/days${N_DAYS}_chance${CC}"
      echo "mkdir -p ${MC_DIR} && ${PYTHON} ${MC_BASELINE} --n-days-ahead ${N_DAYS} --chance-pct ${CC} --data ${DATA} --out ${MC_DIR}/mc_mpc_days${N_DAYS}_chance${CC}.csv"
    done
  done

  # Deterministic MPC baseline: all horizons × forecast modes
  local FORECAST_MODES=("perfect" "zero")
  for N_DAYS in "${N_DAYS_AHEAD_VALS[@]}"; do
    for FM in "${FORECAST_MODES[@]}"; do
      local MPC_DIR="${OUTPUT_DIR}/det_mpc/days${N_DAYS}_${FM}"
      echo "mkdir -p ${MPC_DIR} && ${PYTHON} ${MPC_BASELINE} --n-days-ahead ${N_DAYS} --forecast-mode ${FM} --data ${DATA} --out ${MPC_DIR}/det_mpc_days${N_DAYS}_${FM}.csv"
    done
  done
}

CMDS=$(generate_commands)
N_CMDS=$(echo "${CMDS}" | wc -l)
echo "Launching ${N_CMDS} experiments in parallel..."

# ── Run ──────────────────────────────────────────────────────
MAX_JOBS=48          # all experiments fit on one 192-core node
RUNNING=0

while IFS= read -r CMD; do
  echo "[START] ${CMD}"
  eval "${CMD}" &
  RUNNING=$((RUNNING + 1))
  if [ "${RUNNING}" -ge "${MAX_JOBS}" ]; then
    wait -n            # wait for any one child to finish
    RUNNING=$((RUNNING - 1))
  fi
done <<< "${CMDS}"

wait                    # wait for remaining jobs
echo "All ${N_CMDS} experiments complete."
