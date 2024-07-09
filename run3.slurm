#!/bin/sh

#SBATCH --job-name=DynPETSeg-A3  # Job name
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
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate DynPet

# Declare variable for train or test mode
MODE="test"

if [ "$MODE" = "train" ]; then
    #3 Patients Full Body
    #KINETIC_CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Kinetics_full_body/lucky-firefly-685/epoch=129-step=164840.ckpt
    #All Patients Full Body
    #KINETIC_CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Kinetics_full_body/glorious-flower-706/epoch=99-step=416000.ckpt
    #All Patients Size 192
    KINETIC_CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Kinetics_192/proud-snow-610/epoch=64-step=22555.ckpt

    # Run the script
    python -u ./main.py --experiment seg3 --modality train --kinetic_ckpt $KINETIC_CKPT

elif [ "$MODE" = "test" ]; then
    #FIRST TRAINING
    #-------------------Filtered Liver Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/clean-frog-224/epoch=59-step=18900.ckpt
    #-------------------Aorta Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/peach-durian-239/epoch=9-step=6870.ckpt
    #-------------------Aorta Overfitted Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/peach-durian-239/last.ckpt
    #-------------------Kidneys Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/silver-firefly-242/epoch=29-step=7350.ckpt
    #-------------------Vessels Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/polar-gorge-243/epoch=19-step=16720.ckpt


    #SECOND TRAINING (FOCAL DICE LOSS)
    #-------------------Filtered Liver Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/deep-meadow-316/epoch=9-step=3150.ckpt
    #-------------------Aorta Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/fast-firebrand-312/epoch=9-step=6870.ckpt
    #-------------------Kidneys Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/graceful-jazz-310/epoch=9-step=2450.ckpt
    #-------------------Vessels + 4 Chambers Segmentation-------------------
    #CKPT=/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/checkpoints/amber-sky-320/epoch=9-step=16710.ckpt


    #THIRD TRAINING (MONAI DROPOUT)
    #-------------------All organs Seg slices-------------------------
    #CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Adaptation_3/logical-capybara-677/epoch=29-step=3150.ckpt
    #CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Adaptation_3/logical-capybara-677/last.ckpt
    #-------------------All organs All slices-------------------------
    #CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Adaptation_3/graceful-lake-679/epoch=14-step=3900.ckpt
    CKPT=/home/guests/jorge_padilla/code/DynamicPET_Segmentation/ckpts/Adaptation_3/graceful-lake-679/last.ckpt

    # Run the script in test mode
    python -u ./main.py --experiment seg3 --modality test --ckpt $CKPT
fi

conda deactivate


#Results
#Filtered Liver Segmentation
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/clean-frog-224_best/tables
#Aorta Segmentation
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/peach-durian-239_best/tables
#Aorta Overfitted Segmentation
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/peach-durian-239_last/table
#Kidneys Segmentation
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/silver-firefly-242_best/tables
#Vessels Segmentation
#/home/guests/jorge_padilla/output/models/DynamicPET_Segmentation/output/polar-gorge-243_best/tables
