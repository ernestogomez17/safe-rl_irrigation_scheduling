#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --job-name=irrigation_parallel
#SBATCH --output=/scratch/egomez/irrigation_project_output/irrigation_hyperparameter_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ernesto.gomez@mail.utoronto.ca

#=======================================================================
# 1. SETUP THE JOB ENVIRONMENT
#=======================================================================
echo "✅ Setting up job environment..."
echo "Job running on node $(hostname)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"

# Load necessary modules
module load StdEnv/2023 gcc/12.3

# Activate the virtual environment
source /scratch/egomez/safety-env/bin/activate

#=======================================================================
# 2. GENERATE THE COMMANDS TO BE RUN
#=======================================================================
echo "📝 Generating command file..."

# Define the file to store our commands
COMMANDS_FILE="commands_to_run.txt"

# Clear the file to ensure it's empty before we start
> "$COMMANDS_FILE"

# --- DEFINE YOUR PARAMETERS ---
MODELS=("SACLagrangian" "DDPGLagrangian")
DAYS_AHEAD=(1 3 7)
CHANCE_CONSTRAINTS=(0.75 0.85 0.95 1.0)

# --- NESTED LOOPS TO WRITE EACH COMMAND TO THE FILE ---
for model in "${MODELS[@]}"; do
  for days in "${DAYS_AHEAD[@]}"; do
    for chance in "${CHANCE_CONSTRAINTS[@]}"; do

      # This echo command writes one full line to our text file.
      # Each line is a complete Python command with a unique set of parameters.
      echo "/scratch/egomez/safety-env/bin/python3 /home/egomez/irrigation_project/irrigation_optimization_cluster.py  --model-type ${model} --n-days-ahead ${days} --chance-const ${chance}" >> "$COMMANDS_FILE"

    done
  done
done

echo "Generated $(wc -l < $COMMANDS_FILE) commands in $COMMANDS_FILE"

#=======================================================================
# 3. EXECUTE THE COMMANDS IN PARALLEL
#=======================================================================
echo "🚀 Starting parallel execution of all experiments using GNU Parallel..."

# --jobs tells parallel how many tasks to run simultaneously. Using the Slurm variable is best practice.
# --joblog creates a detailed log of which commands ran, their exit codes, and timing.
parallel --jobs "${SLURM_CPUS_PER_TASK:-24}" --joblog parallel_run.log < "$COMMANDS_FILE"

#=======================================================================
# 4. CLEANUP
#=======================================================================
echo "✅ All experiments are complete."

# Deactivate the virtual environment
deactivate

echo "Job finished."
