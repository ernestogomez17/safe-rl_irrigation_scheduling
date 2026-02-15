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
# 2. CONFIGURATION GRID
#=======================================================================
MODELS=("SACLagrangian" "DDPGLagrangian")
DAYS_AHEAD=(1 3 7)
CHANCE_CONSTRAINTS=(0.75 0.95)

# Compute workers per configuration to fully utilise allocated CPUs.
# 2 models x 3 horizons x 2 chance = 12 configurations.
NUM_COMBOS=$(( ${#MODELS[@]} * ${#DAYS_AHEAD[@]} * ${#CHANCE_CONSTRAINTS[@]} ))
WORKERS_PER_CONFIG=$(( ${SLURM_CPUS_PER_TASK:-192} / NUM_COMBOS ))
[[ $WORKERS_PER_CONFIG -lt 1 ]] && WORKERS_PER_CONFIG=1

TOTAL_PROCS=$(( NUM_COMBOS * WORKERS_PER_CONFIG ))
echo "Configs: ${NUM_COMBOS} | Workers/config: ${WORKERS_PER_CONFIG} | Total processes: ${TOTAL_PROCS}"

#=======================================================================
# 3. LAUNCH — pipe commands directly to GNU Parallel (no temp file)
#=======================================================================
echo "Starting parallel execution..."

generate_commands() {
  for model in "${MODELS[@]}"; do
    for days in "${DAYS_AHEAD[@]}"; do
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
  done
}

generate_commands | parallel --jobs "${TOTAL_PROCS}" \
                             --joblog "${BASE_PATH}/parallel_run_${SLURM_JOB_ID}.log"

#=======================================================================
# 4. DONE
#=======================================================================
deactivate
echo "Job finished at $(date)."
