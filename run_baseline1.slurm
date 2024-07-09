#!/bin/sh

#SBATCH --job-name=DynPETSeg-B1  # Job name
#SBATCH --output=output/logs/DynPETSeg-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=output/logs/DynPETSeg-%A.err  # Standard error of the script
#SBATCH --time=03-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
# SBATCH --gres=gpu:2
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
conda deactivate 
conda activate DynPet

# Declare variable for train or test mode
MODE="test"

if [ "$MODE" = "train" ]; then

    # Run the script
    python -u ./main.py --experiment bas1 --modality train 

elif [ "$MODE" = "test" ]; then

    # Declare variable for trained checkpoint
    #-------------------Filtered Liver Segmentation------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/pleasant-pyramid-225/epoch=39-step=3160.ckpt
    #-------------------Aorta Segmentation---------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/youthful-yogurt-307/epoch=249-step=43000.ckpt
    #-------------------Kidneys Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/silver-flower-247/epoch=9-step=620.ckpt
    #-------------------Vessels Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/true-water-249/epoch=39-step=8360.ckpt
    #-------------------All organs Monai Dropout---------------------
    CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Baseline_1/magic-flower-624/epoch=119-step=62400.ckpt
    #CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Baseline_1/magic-flower-624/last.ckpt

    # Run the script in test mode
    python -u ./main.py --experiment bas1 --modality test --ckpt $CKPT
fi

conda deactivate

#Results
#-------------------Filtered Liver Segmentation------------
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/pleasant-pyramid-225_best/tables
#-------------------Aorta Segmentation---------------------
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/youthful-yogurt-307_best/tables
#-------------------Kidneys Segmentation-------------------
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/silver-flower-247_best/tables
#-------------------Vessels Segmentation-------------------
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/true-water-249_best/tables