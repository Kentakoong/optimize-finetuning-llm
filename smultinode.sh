#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script

module restore
module load Mamba
module load cudatoolkit/23.3_11.8
module load gcc/10.3.0

conda deactivate
conda activate deeptransformers

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
THEID=$(echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]")
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID
echo -------------------------

export NCCL_TIMEOUT=3600000
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=" /project/lt999001-intern/wongkraiwich/working/wk1/finetune/.cache"

accelerate launch \
    --num_processes $((4 * $COUNT_NODE)) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision fp16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --dynamo_backend inductor \
    scripts/train.py \
    --model_name_or_path /project/lt999001-intern/shared/models/Llama-2-13b-chat-hf \
    --train_file /project/lt999001-intern/shared/datasets/alpaca_json/alpaca_train.json \
    --validation_file /project/lt999001-intern/shared/datasets/alpaca_json/alpaca_train.json \
    --seed 42 \
    --max_seq_length 1300 \
    --output_dir /project/lt999001-intern/wongkraiwich/working/wk1/finetune/checkpoint/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_steps 700 \
    --save_total_limit 5 \
    --learning_rate 8e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 1 \
    --deepspeed ./deepspeed_config/deepspeed_3.json \
    --gradient_checkpointing True \
    --tf32 True

#qwen: 1100
#llama: 1300
