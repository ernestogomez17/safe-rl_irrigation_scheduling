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
module load gnu-parallel

# ── Verify ───────────────────────────────────────────────────
cd "${PROJECT_DIR}"
echo "Python: ${PYTHON}"
${PYTHON} --version

# ── Generate commands (30 RL + 12 baseline = 42 total) ──────
generate_commands() {
  local N_DAYS_AHEAD_VALS=(1 3 7)
  local CHANCE_CONST_VALS=(0.75 0.85 0.95 1)
  local LAGRANGIAN_MODELS=("SACLagrangian" "DDPGLagrangian")
  local REGULAR_MODELS=("DDPG" "SAC")
  local BASELINE="${PROJECT_DIR}/models/mc_irrigation_baseline.py"
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
      echo "${PYTHON} ${BASELINE} --n-days-ahead ${N_DAYS} --chance-pct ${CC} --data ${DATA} --out ${OUTPUT_DIR}/mc_baseline_days${N_DAYS}_chance${CC}.csv"
    done
  done
}

echo "Launching $(generate_commands | wc -l) experiments in parallel..."

# ── Run ──────────────────────────────────────────────────────
generate_commands | parallel \
  --jobs ${SLURM_CPUS_PER_TASK:-192} \
  --joblog "${OUTPUT_DIR}/parallel_log_${SLURM_JOB_ID}.txt"

echo "All experiments complete."
