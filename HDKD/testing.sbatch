#!/bin/bash 

#SBATCH --job-name=hdkd_from_base_teacher_200   # Job name 
#SBATCH --output=hdkd_from_base_teacher_200-%j.out    # Standard output and error log 
#SBATCH --qos=l40
#SBATCH --partition=l40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 

python main.py --batch-size 32 --model HDKD  --epochs 200 --teacher-path /home/micml/25m0764/MIC_project/HDKD/weighted_loss_data_models/teacher_model/teacher_model_weighted_loss.pth --data-path /home/micml/25m0764/MIC_project/dataset_dbn/dataset_200