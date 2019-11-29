#!/bin/bash
#Run this script via sbatch:
#sbatch ./script.sh


#Reserve 4 GPUs on the cluster.
#SBATCH -t 24:00:00  -n32 -N1 --gres=gpu:k40:2 --tasks-per-node=32 --distribution=cyclic


#launch 4 copies of the python code, passing the hostname.
srun -n 2  ./ml_parallel.py -m  $HOSTNAME

