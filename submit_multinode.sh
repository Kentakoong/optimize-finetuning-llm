#!/bin/bash
#SBATCH -p gpu  --exclusive             # Specify partition [Compute/Memory/GPU]
#SBATCH -N 8 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt999001                     # Specify project name
#SBATCH -J finetune_test                # Specify job name
#SBATCH -o logs/finetune-%j.out         # Specify output file

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=16
export NCCL_NSOCKS_PERTHREAD=16

START=$(date)
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo go $COUNT_NODE
echo $HOSTNAMES

srun sh smultinode.sh
