#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --job-name=irrigation_hpo
#SBATCH --output=/scratch/egomez/irrigation_hpo/job_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ernesto.gomez@mail.utoronto.ca

#=======================================================================
# 1. SETUP
#=======================================================================
echo "Setting up job environment..."
echo "Node: $(hostname)  |  CPUs: $SLURM_CPUS_PER_TASK"

module load StdEnv/2023 gcc/12.3
module load gnu-parallel
source /home/egomez/irrigation_project/irrigation_env/bin/activate

PYTHON=/home/egomez/irrigation_project/irrigation_env/bin/python3
SCRIPT=/home/egomez/irrigation_project/hpo.py
BASE_PATH=/scratch/egomez/irrigation_hpo
mkdir -p "${BASE_PATH}"

#=======================================================================
# 2. MISSING CONFIGURATIONS (model  days_ahead)
#=======================================================================
# Only the combos whose PID gains are still missing from the table:
#   DDPGLagrangian  d=3
#   DDPGLagrangian  d=7
#   SACLagrangian   d=7
COMBOS=(
  "DDPGLagrangian 3"
  "DDPGLagrangian 7"
  "SACLagrangian  7"
)
CHANCE_CONSTRAINTS=(0.95)

# 3 model-day pairs x 1 chance constraint = 3 configurations.
NUM_COMBOS=$(( ${#COMBOS[@]} * ${#CHANCE_CONSTRAINTS[@]} ))
WORKERS_PER_CONFIG=$(( ${SLURM_CPUS_PER_TASK:-192} / NUM_COMBOS ))
[[ $WORKERS_PER_CONFIG -lt 1 ]] && WORKERS_PER_CONFIG=1

TOTAL_PROCS=$(( NUM_COMBOS * WORKERS_PER_CONFIG ))
echo "Configs: ${NUM_COMBOS} | Workers/config: ${WORKERS_PER_CONFIG} | Total processes: ${TOTAL_PROCS}"

#=======================================================================
# 3. LAUNCH — pipe commands directly to GNU Parallel (no temp file)
#=======================================================================
echo "Starting parallel execution..."

generate_commands() {
  for combo in "${COMBOS[@]}"; do
    read -r model days <<< "${combo}"
    for chance in "${CHANCE_CONSTRAINTS[@]}"; do
      for wid in $(seq 0 $(( WORKERS_PER_CONFIG - 1 ))); do
        echo "${PYTHON} ${SCRIPT} \
          --model-type ${model} \
          --n-days-ahead ${days} \
          --chance-const ${chance} \
          --n-workers ${WORKERS_PER_CONFIG} \
          --worker-id ${wid} \
          --base-path ${BASE_PATH}"
      done
    done
  done
}

generate_commands | parallel --jobs "${TOTAL_PROCS}" \
                             --joblog "${BASE_PATH}/parallel_run_${SLURM_JOB_ID}.log"

#=======================================================================
# 4. DONE
#=======================================================================
deactivate
echo "Job finished at $(date)."
