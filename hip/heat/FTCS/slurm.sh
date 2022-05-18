#!/bin/bash
#SBATCH --job-name=heat
#SBATCH --output=heat%j.out
#SBATCH --error=heat%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=sireeves@ucsc.edu
#SBATCH --partition=gpuq             # queue for job submission
#SBATCH --account=gpuq               # queue for job submission

srun ./heat.exe

