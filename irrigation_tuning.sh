#!/bin/bash
#SBATCH --time=12:00:00
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
echo "Node: $(hostname)  |  CPUs: $SLURM_CPUS_PER_TASK  |  Start: $(date)"

module load StdEnv/2023 gcc/12.3
source /home/egomez/irrigation_project/irrigation_env/bin/activate

PYTHON=/home/egomez/irrigation_project/irrigation_env/bin/python3
SCRIPT=/home/egomez/irrigation_project/hpo.py
BASE_PATH=/scratch/egomez/irrigation_hpo
mkdir -p "${BASE_PATH}"

#=======================================================================
# 2. ALL LAGRANGIAN CONFIGURATIONS (model  days_ahead)
#=======================================================================
COMBOS=(
  "DDPGLagrangian 1"
  "DDPGLagrangian 3"
  "DDPGLagrangian 7"
  "SACLagrangian  1"
  "SACLagrangian  3"
  "SACLagrangian  7"
)
CHANCE_CONSTRAINTS=(0.95)

# ── Parallelism budget ────────────────────────────────────────────────
# 8 workers/config × 6 configs = 48 concurrent Python processes.
# Each worker trains 100-epoch agents, so 48 is enough to keep the node
# busy without oversubscribing memory.  The remaining cores service the
# NumPy/PyTorch thread pools inside each process (threads=1 per worker,
# but OS scheduling benefits from headroom).
WORKERS_PER_CONFIG=8

NUM_COMBOS=$(( ${#COMBOS[@]} * ${#CHANCE_CONSTRAINTS[@]} ))
TOTAL_PROCS=$(( NUM_COMBOS * WORKERS_PER_CONFIG ))
echo "Configs: ${NUM_COMBOS} | Workers/config: ${WORKERS_PER_CONFIG} | Total processes: ${TOTAL_PROCS}"

#=======================================================================
# 3. LAUNCH — shell backgrounding (resumes from existing journals)
#=======================================================================
echo "Starting parallel execution..."

MAX_CONCURRENT=${TOTAL_PROCS}
running=0

for combo in "${COMBOS[@]}"; do
  read -r model days <<< "${combo}"
  for chance in "${CHANCE_CONSTRAINTS[@]}"; do
    for wid in $(seq 0 $(( WORKERS_PER_CONFIG - 1 ))); do
      echo "  Launching: ${model} d=${days} chance=${chance} worker=${wid}"
      ${PYTHON} ${SCRIPT} \
        --model-type ${model} \
        --n-days-ahead ${days} \
        --chance-const ${chance} \
        --n-workers ${WORKERS_PER_CONFIG} \
        --worker-id ${wid} \
        --base-path ${BASE_PATH} &

      running=$(( running + 1 ))
      if [[ $running -ge $MAX_CONCURRENT ]]; then
        wait -n
        running=$(( running - 1 ))
      fi
    done
  done
done

# Wait for all remaining background jobs
wait
echo "All HPO workers finished."

#=======================================================================
# 4. DONE
#=======================================================================
deactivate
echo "Job finished at $(date)."
