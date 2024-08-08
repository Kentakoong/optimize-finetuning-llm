#!/bin/bash
#SBATCH -p gpu-devel --exclusive               # Specify partition [Compute/Memory/GPU]
#SBATCH -c 64                            # Specify number of processors per task
#SBATCH --ntasks-per-node=1		         # Specify number of tasks per node
#SBATCH --gpus-per-node=4		         # Specify total number of GPUs
#SBATCH -t 1:00:00                       # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt999001                      # Specify project name
#SBATCH -J offload                 # Specify job name
#SBATCH -o ./logs/finetune-%j.out        # Specify output file

NTHREADS="4"
PTHREADS="4"

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --nthreads)
        NTHREADS="$2"
        shift 2
        ;;
    --pthreads)
        PTHREADS="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

# Use the variables as needed
echo "NTHREADS: $NTHREADS"
echo "PTHREADS: $PTHREADS"

START=$(date)
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export SLURM_PROCID=SLURM_PROCID

echo go $COUNT_NODE
echo $HOSTNAMES

if [[ -z "$NTHREADS" || -z "$PTHREADS" ]]; then
    export LOG_DIR="./logs/finetune-${SLURM_JOB_ID}"
else
    export LOG_DIR="./logs/finetune-${SLURM_JOB_ID}-nthreads-${NTHREADS}-pthreads-${PTHREADS}"
fi

mkdir -p $LOG_DIR
mkdir -p $LOG_DIR/node_log

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=$NTHREADS
export NCCL_NSOCKS_PERTHREAD=$PTHREADS
export NCCL_DEBUG_FILE=${LOG_DIR}/nccl-${SLURM_JOB_ID}.log
export NCCL_TOPO_DUMP_FILE=${LOG_DIR}/nccl-topo-${SLURM_JOB_ID}.log

srun --output=${LOG_DIR}/node_log/node-%t.out sh smultinode.sh
