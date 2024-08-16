#!/bin/bash
#SBATCH -p gpu --exclusive               # Specify partition [Compute/Memory/GPU]
#SBATCH -c 64                            # Specify number of processors per task
#SBATCH --ntasks-per-node=1		         # Specify number of tasks per node
#SBATCH --gpus-per-node=4		         # Specify total number of GPUs
#SBATCH -t 1:00:00                       # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt999001                      # Specify project name
#SBATCH -J scaling                       # Specify job name
#SBATCH -o ./logs/finetune-%j.out        # Specify output file

module restore
module load Mamba
module load PrgEnv-gnu
module load cpe-cuda/23.03
module load cudatoolkit/23.3_11.8

if [ -f pre-submit.sh ]; then
    source pre-submit.sh
fi

: "${NTHREADS:=8}"
: "${PTHREADS:=2}"
: "${BATCH_SIZE:=4}"
: "${DEEPSPEED_STAGE:=2}"
: "${MODEL_SIZE:=7b}"
: "${TASK:=finetune}"
: "${RUN_WITH:=conda}"
: "${ENV_PATH:=}"

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
    --batch_size)
        BATCH_SIZE="$2"
        shift 2
        ;;
    --deepspeed_stage)
        DEEPSPEED_STAGE="$2"
        shift 2
        ;;
    --model_size)
        MODEL_SIZE="$2"
        shift 2
        ;;
    --task)
        TASK="$2"
        shift 2
        ;;
    --run_with)
        RUN_WITH="$2"
        shift 2
        ;;
    --env_path)
        ENV_PATH="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

if [ "$ENV_PATH" == "" ]; then
    echo "ENV_PATH is not set, please set the path to the environment using --env_path"
    exit 1
fi

if [ "$PROJ_PATH" == "" ]; then
    echo "PROJ_PATH is not set, please export the path to the project directory"
    exit 1
elif [ "$SHARED_PATH" == "" ]; then
    echo "SHARED_PATH is not set, please export the path to the shared directory"
    exit 1
elif [ "$CACHE_PATH" == "" ]; then
    echo "CACHE_PATH is not set, please export the path to the cache directory"
    exit 1
elif [ "$ENV_PATH" == "" ]; then
    echo "ENV_PATH is not set, please export the path to the environment"
    exit 1
fi

conda deactivate
conda activate $ENV_PATH

export WANDB_MODE="offline"
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

if [ "$TASK" == "nccl" ]; then
    LOG_DIR="./logs/${NTHREADS}nth-${PTHREADS}pth-${SLURM_JOB_ID}" # for nccl testing
elif [ "$TASK" == "scaling" ]; then
    LOG_DIR="../scaling/stage-${DEEPSPEED_STAGE}/llama-${MODEL_SIZE}/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}" # for scaling
else 
    LOG_DIR="./logs/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}"
fi

mkdir -p $LOG_DIR/node_log

export LOG_DIR=$LOG_DIR

export NCCL_TIMEOUT=3600000
export NCCL_DEBUG=DEBUG
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=$NTHREADS
export NCCL_NSOCKS_PERTHREAD=$PTHREADS
export NCCL_DEBUG_FILE=${LOG_DIR}/nccl-${SLURM_JOB_ID}.log
export NCCL_TOPO_DUMP_FILE=${LOG_DIR}/nccl-topo-${SLURM_JOB_ID}.log
export BATCH_SIZE=$BATCH_SIZE
export DEEPSPEED_STAGE=$DEEPSPEED_STAGE
export MODEL_SIZE=$MODEL_SIZE

export PROJ_PATH="/project/lt999001-intern/wongkraiwich/working/llm/finetune"
export SHARED_PATH="/scratch/lt999001-intern/shared"
export CACHE_PATH="/scratch/lt999001-intern/.cache"
export ENV_PATH="/project/lt999001-intern/wongkraiwich/env/deeptransformers"

export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=$CACHE_PATH
export HF_HUB_CACHE=$CACHE_PATH
export HF_HOME=$CACHE_PATH
export HF_DATASETS_CACHE=$CACHE_PATH
export TORCH_HOME=$CACHE_PATH
export XDG_CACHE_HOME=$CACHE_PATH
export TRITON_CACHE_DIR=$CACHE_PATH

echo -------ENVIRONMENT-------
echo Python Path: $(which python)
echo Batch Size: $BATCH_SIZE
echo Deepspeed Stage: $DEEPSPEED_STAGE
echo Model Size: $MODEL_SIZE
echo -------------------------
echo NTHREADS: $NTHREADS
echo PTHREADS: $PTHREADS
echo NODES: $COUNT_NODE
echo HOSTNAMES: $HOSTNAMES
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo -------------------------

srun --output=${LOG_DIR}/node_log/node-%t.out sh submit-node.sh
