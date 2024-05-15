#!/bin/bash


#SBATCH --job-name="gkit_cahcing"
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=16GB
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python/3.8.12
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-matplotlib
module load py-pandas
module load py-scikit-learn
module load py-pip

python -m pip install --user cvxpy
python -m pip install --user statsmodels

srun python main.py > pi.log
