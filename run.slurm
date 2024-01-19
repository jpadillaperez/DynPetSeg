#!/bin/sh

#SBATCH --job-name=DynPETSeg  # Job name
#SBATCH --output=output/logs/DynPETSeg-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/DynPETSeg-%A.err  # Standard error of the script
#SBATCH --time=01-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1
#set gpu memory limit
#SBATCH --mem=46G  # Memory limit
#SBATCH --cpus-per-task=8  # Number of CPU cores per task

CURRENT_LOG_DIR=output/logs
OLD_LOG_DIR=output/logs/previous_logs

# Move old logs to the new directory except for the current log file
for file in $CURRENT_LOG_DIR/*.out $CURRENT_LOG_DIR/*.err; do
    if [[ $file != $CURRENT_LOG_DIR/DynPETSeg-$SLURM_JOB_ID.out ]] && [[ $file != $CURRENT_LOG_DIR/DynPETSeg-$SLURM_JOB_ID.err ]]; then
        mv "$file" "$OLD_LOG_DIR/"
    fi
done

# load python module
source /home/guests/jorge_padilla/miniconda3/etc/profile.d/conda.sh

# Set your wandb API key
export WANDB_API_KEY=$(cat ./wandb_api_key.txt)

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate DynPet

python -u ./main.py

conda deactivate