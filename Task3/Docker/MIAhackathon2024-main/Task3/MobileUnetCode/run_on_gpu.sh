#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --time=8:00:00
# module load python/3.11.3
python3 --version
source ../envs/hackathon_env/bin/activate
echo 'working directory ...'
pwd
# carbontracker python3 train_mobilenet.py --logdir="carbontracker_logs"
./train.sh