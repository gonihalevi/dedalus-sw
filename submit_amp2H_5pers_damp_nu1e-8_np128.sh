#!/bin/bash

#SBATCH -J amp2H_5pers_damp_nu1e-8_np128
#SBATCH -o amp2H_5pers_damp_nu1e-8_np128.out
#SBATCH -N 5 
#SBATCH -n 140
#SBATCH -t 48:00:00
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=ghalevi@princeton.edu

source /home/ghalevi/dedalus/startup.sh
mpirun -np 128 python /home/ghalevi/dedalus/dedalus/dedalus_sphere/amp2H_5pers_damp_nu1e-8_np128.py
