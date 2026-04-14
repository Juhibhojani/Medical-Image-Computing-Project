#!/bin/bash 

#SBATCH --job-name=training_hdkd_full     # Job name 
#SBATCH --output=training_hdkd_full-%j.out    # Standard output and error log 
#SBATCH --qos=a40
#SBATCH --partition=a40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 

python main.py --batch-size 32 --model HDKD  --epochs 200 --teacher-path /home/micml/25m0764/MIC_project/HDKD/4200_data_models/teacher_model/teacher_600_smote_model.pth --data-path /home/micml/25m0764/MIC_project/final_dataset/dataset_600