#!/bin/bash 

#SBATCH --job-name=teacher_model_with_weighted_loss    # Job name 
#SBATCH --output=teacher_model_with_weighted_loss-%j.out    # Standard output and error log 
#SBATCH --qos=a40
#SBATCH --partition=a40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 

python main.py --batch-size 32 --model teacher_model --epochs 200 --data-path /home/micml/25m0764/MIC_project/final_dataset/dataset_600