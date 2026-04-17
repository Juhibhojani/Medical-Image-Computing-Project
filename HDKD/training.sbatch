#!/bin/bash 
#SBATCH --job-name=fine_tuning_teacher_config2_200   # Job name 
#SBATCH --output=fine_tuning_teacher_config2_200-%j.out    # Standard output and error log 
#SBATCH --qos=l40
#SBATCH --partition=l40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 
python main_finetuning.py --batch-size 32 --model teacher_model  --epochs 200 --data-path /home/micml/25m0764/MIC_project/dataset_dbn/dataset_200
