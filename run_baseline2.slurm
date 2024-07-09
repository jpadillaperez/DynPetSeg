#!/bin/sh

#SBATCH --job-name=DynPETSeg-B1  # Job name
#SBATCH --output=output/logs/DynPETSeg-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/DynPETSeg-%A.err  # Standard error of the script
#SBATCH --time=03-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:2
# SBATCH --gres=gpu:1
#SBATCH --mem=46G  # Memory limit
#SBATCH --cpus-per-task=12  # Number of CPU cores per task

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
conda deactivate 
conda activate DynPet

# Declare variable for train or test mode
MODE="test"

if [ "$MODE" = "train" ]; then

    # Run the script
    python -u ./main.py --experiment bas2 --modality train 

elif [ "$MODE" = "test" ]; then

    #CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Baseline_2/dulcet-firefly-662/epoch=59-step=15600.ckpt
    CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Baseline_2/dulcet-firefly-662/last.ckpt

    # Run the script in test mode
    python -u ./main.py --experiment bas2 --modality test --ckpt $CKPT
fi

conda deactivate