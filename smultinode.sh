#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

module restore
module load Mamba
module load PrgEnv-gnu
module load cpe-cuda/23.03
module load cudatoolkit/23.3_11.8
module use /project/lt999001-intern/.local/easybuild/modules/all
module load libaio/0.3.113

# module load cudatoolkit/23.3_11.8
module load gcc


conda deactivate
conda activate ./env-aio

echo -------ds_report---------
ds_report

echo -------ENVIRONMENT-------
echo myuser=$(whoami)
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc $(which mpicc)
echo HOSTNAMES = $HOSTNAMES
echo hostname = $(hostname)
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=$(hostname)
echo SLURM_PROCID=$SLURM_PROCID
echo -------------------------

echo PWD=$(pwd)
echo LS=$(ls)

export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=".cache"
export HF_HUB_CACHE=".cache"
export HF_HOME=".cache"
export HF_DATASETS_CACHE=".cache"
export TORCH_HOME=".cache"
export XDG_CACHE_HOME=".cache"
export TRITON_CACHE_DIR=".cache"
export BNB_CUDA_VERSION=118

export LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/11.8/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/math_libs/11.8/lib64:$LD_LIBRARY_PATH"



# export CUDA_HOME="/usr/local/cuda"

echo ------DEEPSPEED--------
cat "deepspeed_config/deepspeed_infinity.json"
echo -----------------------

# PROJ_PATH=/project/lt999001-intern/wongkraiwich/working/llm/finetune
# SHARED_PATH=/scratch/lt999001-intern/shared

accelerate launch \
    --num_processes $((4 * $COUNT_NODE)) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision bf16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --dynamo_backend inductor \
    scripts/train.py \
    --pretrained_model_name_or_path /scratch/lt999001-intern/shared/models/Llama-2-13b-chat-hf \
    --train_file /scratch/lt999001-intern/shared/datasets/alpaca_json/alpaca_all.json \
    --validation_file /project/lt999001-intern/shared/datasets/alpaca_json/alpaca_validation.json \
    --seed 42 \
    --max_seq_length 1300 \
    --output_dir /project/lt999001-intern/checkpoint/full_check \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 8e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --deepspeed ./deepspeed_config/deepspeed_infinity.json \
    --gradient_checkpointing True \
    --tf32 True \
    --bf16 True \
    --dataloader_num_workers 16 \
    --ddp_find_unused_parameters False \
    --log_dir $LOG_DIR \
    --node_number $SLURM_PROCID
#qwen: 1100
#llama: 1300
