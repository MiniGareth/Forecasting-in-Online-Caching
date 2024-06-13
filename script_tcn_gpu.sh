#!/bin/bash


#SBATCH --job-name="gkit_cahcing"
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load cuda/11.6
module load python/3.8.12
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-matplotlib
module load py-pandas
module load py-scikit-learn
module load py-pip
module load py-torch

python -m pip install --user cvxpy
python -m pip install --user statsmodels
python -m pip install --user tensorboard

srun python tcn_main.py > tcn_main_gpu_mse.log
