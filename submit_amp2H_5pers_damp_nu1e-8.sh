#!/bin/bash

#SBATCH -J amp2H_5pers_damp_nu1e-8
#SBATCH -o amp2H_5pers_damp_nu1e-8.out
#SBATCH -N 2 
#SBATCH -n 28
#SBATCH -t 48:00:00
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=ghalevi@princeton.edu

source /home/ghalevi/dedalus/startup.sh
mpirun -np 32 python /home/ghalevi/dedalus/dedalus/dedalus_sphere/amp2H_5pers_damp_nu1e-8.py
