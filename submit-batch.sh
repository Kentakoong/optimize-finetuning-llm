#!/bin/bash
#SBATCH -p gpu --exclusive               # Specify partition [Compute/Memory/GPU]
#SBATCH --exclude=lanta-g-[159-174]      # Exclude nodes
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
# : "${ENV_PATH:=}"
: "${SCALING_TYPE:=}"
: "${WO_LORA:=NO}"

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
    # --env_path)
    #     ENV_PATH="$2"
    #     shift 2
    #     ;;
    --scaling_type)
        SCALING_TYPE="$2"
        shift 2
        ;;
    --wo_lora)
        WO_LORA="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

if [ "$PROJ_PATH" == "" ]; then
    echo "PROJ_PATH is not set, please export the path to the project directory"
    exit 1
fi
if [ "$SHARED_PATH" == "" ]; then
    echo "SHARED_PATH is not set, please export the path to the shared directory"
    exit 1
fi
if [ "$CACHE_PATH" == "" ]; then
    echo "CACHE_PATH is not set, please export the path to the cache directory"
    exit 1
fi
# if [ "$ENV_PATH" == "" ]; then
#     echo "ENV_PATH is not set, please export the path to the environment"
#     exit 1
# fi

conda deactivate

export WANDB_MODE="offline"
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

# if [ "$TASK" == "nccl" ]; then
#     LOG_DIR="./logs/${NTHREADS}nth-${PTHREADS}pth-${SLURM_JOB_ID}" # for nccl testing
# elif [ "$TASK" == "scaling" ]; then
#     folstru=""
#     if [ "$WO_LORA" == "YES" ]; then
#         folstru="/wo-lora"
#     fi
#     if [ "$SCALING_TYPE" != "" ]; then
#         LOG_DIR="../scaling$folstru/${SCALING_TYPE}/stage-${DEEPSPEED_STAGE}/llama-${MODEL_SIZE}/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}"
#     else 
#         LOG_DIR="../scaling/stage-${DEEPSPEED_STAGE}/llama-${MODEL_SIZE}/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}" 
#     fi
# else 
#     LOG_DIR="./logs/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}"
# fi

folstru=""
if [ "$WO_LORA" == "YES" ]; then
    folstru="/wo-lora"
fi

LOG_DIR="../test-env/stage-${DEEPSPEED_STAGE}/llama-${MODEL_SIZE}/${COUNT_NODE}n-${BATCH_SIZE}b-${SLURM_JOB_ID}"

if [ "$WO_LORA" == "YES" ]; then
    export filename="train_wo_lora.py"
else
    export filename="train.py"
fi

mkdir -p $LOG_DIR/node_log
mkdir -p $PROJ_PATH/checkpoint/$SLURM_JOB_ID

export LOG_DIR=$LOG_DIR

export NCCL_TIMEOUT=3600000
export NCCL_DEBUG=DEBUG
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=$NTHREADS
export NCCL_NSOCKS_PERTHREAD=$PTHREADS
export NCCL_DEBUG_FILE=${LOG_DIR}/nccl-${SLURM_JOB_ID}.log
export NCCL_TOPO_DUMP_FILE=${LOG_DIR}/nccl-topo-${SLURM_JOB_ID}.log

export FI_MR_CACHE_MONITOR=userfaultd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256

export BATCH_SIZE=$BATCH_SIZE
export DEEPSPEED_STAGE=$DEEPSPEED_STAGE
export MODEL_SIZE=$MODEL_SIZE

export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=$CACHE_PATH
export HF_HUB_CACHE="$CACHE_PATH/huggingface"
export HF_HOME="$CACHE_PATH/huggingface"
export HF_DATASETS_CACHE="$CACHE_PATH/huggingface"
export TORCH_HOME=$CACHE_PATH
export XDG_CACHE_HOME=$CACHE_PATH
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1

echo -------ENVIRONMENT-------
echo Python Path: $(which python)
echo Batch Size: $BATCH_SIZE
echo Deepspeed Stage: $DEEPSPEED_STAGE
echo Model Size: $MODEL_SIZE
echo Train with LoRA: $WO_LORA
echo -------------------------
echo NTHREADS: $NTHREADS
echo PTHREADS: $PTHREADS
echo NODES: $COUNT_NODE
echo HOSTNAMES: $HOSTNAMES
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo -------------------------

srun --output=${LOG_DIR}/node_log/node-%t.out sh submit-node.sh
