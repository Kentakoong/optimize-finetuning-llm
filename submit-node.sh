#!/bin/bash

echo "------INFO--------"
echo node_number: $SLURM_PROCID
echo hostname: $(hostname)
echo "------------------"
echo ""

echo "----LAUNCHING TRAINING----"

if [ "$WO_LORA" == "YES" ]; then
    echo "Training without LoRA"
    filename="train_wo_lora.py"
else
    echo "Training with LoRA"
    filename="train.py"
fi

accelerate launch \
    --num_processes $((4 * $COUNT_NODE)) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision bf16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --dynamo_backend inductor \
    $PROJ_PATH/scripts/$filename \
        --pretrained_model_name_or_path "$SHARED_PATH/models/Llama-2-$MODEL_SIZE-chat-hf" \
        --train_file $SHARED_PATH/datasets/alpaca_json/alpaca_all.json \
        --validation_file $SHARED_PATH/datasets/alpaca_json/alpaca_validation.json \
        --seed 42 \
        --max_seq_length 1300 \
        --output_dir $PROJ_PATH/checkpoint \
        --num_train_epochs 1 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --save_steps 700 \
        --save_total_limit 5 \
        --learning_rate 8e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type cosine \
        --gradient_accumulation_steps 1 \
        --deepspeed "$PROJ_PATH/deepspeed_config/deepspeed_$DEEPSPEED_STAGE.json" \
        --gradient_checkpointing True \
        --tf32 True \
        --bf16 True \
        --fp16 False \
        --max_grad_norm 1.0 \
        --logging_steps 10 \
        --dataloader_num_workers 16 \
        --ddp_find_unused_parameters False \
        --log_dir $LOG_DIR \
        --node_number $SLURM_PROCID
