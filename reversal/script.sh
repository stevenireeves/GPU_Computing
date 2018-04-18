#!/bin/bash
#SBATCH --job-name=reverse
#SBATCH --output=reverse%j.out
#SBATCH --error=reverse%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=sireeves@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

srun ./reversal.exe

