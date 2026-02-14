#!/bin/bash  
#SBATCH --time=45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --job-name=irrigation_project
#SBATCH --output=/scratch/egomez/irrigation_project_output/irrigation_parallel_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ernesto.gomez@mail.utoronto.ca

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

COMMAND_FILE="commands_irrigation.txt"
> "$COMMAND_FILE" # Clear the file to ensure it's fresh for this run

echo "📝 Creating command list for 30 experiments..."

# Define hyperparameter values
N_DAYS_AHEAD_VALS=(1 3 7)
CHANCE_CONST_VALS_LAGRANGIAN=(0.75 0.85 0.95 1)
LAGRANGIAN_MODELS=("SACLagrangian" "DDPGLagrangian")
REGULAR_MODELS=("DDPG" "SAC")

# Generate commands for Lagrangian models
for MODEL_TYPE in "${LAGRANGIAN_MODELS[@]}"; do
  for N_DAYS_AHEAD in "${N_DAYS_AHEAD_VALS[@]}"; do
    for CHANCE_CONST in "${CHANCE_CONST_VALS_LAGRANGIAN[@]}"; do
      echo "python /home/egomez/irrigation_project/experiments.py --n_days_ahead $N_DAYS_AHEAD --chance_const $CHANCE_CONST --model_type $MODEL_TYPE" >> "$COMMAND_FILE"
    done
  done
done

# Generate commands for regular models
for MODEL_TYPE in "${REGULAR_MODELS[@]}"; do
  for N_DAYS_AHEAD in "${N_DAYS_AHEAD_VALS[@]}"; do
    echo "python /home/egomez/irrigation_project/experiments.py --n_days_ahead $N_DAYS_AHEAD --chance_const 1 --model_type $MODEL_TYPE" >> "$COMMAND_FILE"
  done
done

echo "✅ Command list created. Found $(wc -l < "$COMMAND_FILE") commands."

# --- STEP 2: EXECUTE COMMANDS IN PARALLEL ---

echo "🚀 Launching experiments in parallel using GNU Parallel..."

# Load necessary modules
module load StdEnv/2023
module load gnu-parallel

# Activate the virtual environment
source /scratch/egomez/irrigation_env/bin/activate

# Run all commands from the file, using all allocated cores.
# The --joblog option is highly recommended as it keeps a useful record of which jobs ran,
# their runtime, and their exit status, which is great for debugging.
parallel --joblog parallel_log.txt < "$COMMAND_FILE"

# Deactivate the virtual environment
deactivate

echo "✅ All experiments are complete."
