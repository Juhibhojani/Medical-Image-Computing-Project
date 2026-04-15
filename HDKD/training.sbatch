#!/bin/bash 

#SBATCH --job-name=transformer_based_teacher_full_weighted    # Job name 
#SBATCH --output=transformer_based_teacher_full_weighted-%j.out    # Standard output and error log 
#SBATCH --qos=l40
#SBATCH --partition=l40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 
python main.py --batch-size 32 --model student_model  --epochs 200 --data-path /home/micml/25m0764/MIC_project/final_dataset/dataset_600