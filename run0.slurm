#!/bin/sh

#SBATCH --job-name=DynPETSeg-A0  # Job name
#SBATCH --output=output/logs/DynPETSeg-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/DynPETSeg-%A.err  # Standard error of the script
#SBATCH --time=03-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1
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
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate DynPet

#Declare variable for kinetic checkpoint
#3 Patients
#KINETIC_CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/lucky-firefly-685/epoch=129-step=164840.ckpt
#All Patients
#KINETIC_CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/glorious-flower-706/epoch=99-step=416000.ckpt

# Run the script
#python -u ./main.py --experiment seg0 --modality train --kinetic_ckpt $KINETIC_CKPT
python -u ./main.py --experiment seg0 --modality train

conda deactivate