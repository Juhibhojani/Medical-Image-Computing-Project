#!/bin/bash 

#SBATCH --job-name=testing_hdkd_350     # Job name 
#SBATCH --output=testing_hdkd_350-%j.out    # Standard output and error log 
#SBATCH --qos=a40
#SBATCH --partition=a40 
#SBATCH --cpus-per-task=8 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --output=logs/%x_%j.out 
#SBATCH --error=logs/%x_%j.err 

python test_modified.py --model HDKD --checkpoint /home/micml/25m0764/MIC_project/HDKD/4200_data_models/hdkd_350/student_350_model.pth --batch-size 16 --data-path /home/micml/25m0764/MIC_project/final_dataset/dataset_600/test