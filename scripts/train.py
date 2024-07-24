import inspect

import pandas as pd
from llm_finetune.arguments import (DataArguments, LoggingArguments,
                                    ModelArguments, TrainingArguments)
from llm_finetune.dataset import make_supervised_data_module
from llm_finetune.trainer import EpochTimingCallback
from nvitop import ResourceMetricCollector
from peft import LoraConfig
from torch import bfloat16
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser, set_seed)
from trl import SFTConfig, SFTTrainer


def extract_dict(obj):
    """Extracts a dictionary from an object."""
    if hasattr(obj, '__dict__'):
        return extract_dict(obj.__dict__)
    else:
        return obj


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bfloat16,
        bnb_4bit_quant_storage=bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype=bfloat16,
        use_cache=False,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    tokenizer_config = extract_dict(tokenizer)

    # print("-------------------TokenizerConfig-------------------")
    # print(tokenizer_config)

    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
    )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    # Get the signature of SFTConfig.__init__
    sft_config_signature = inspect.signature(SFTConfig.__init__)
    sft_config_params = sft_config_signature.parameters

    # Filter out unexpected arguments
    filtered_args = {k: v for k, v in vars(
        training_args).items() if k in sft_config_params}

    # Explicitly pass all training arguments
    config = SFTConfig(**filtered_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        **data_module,
        callbacks=[EpochTimingCallback()],
        peft_config=peft_config,
        packing=False,
    )

    trainer_config = extract_dict(trainer)

    print("-------------------TrainerConfig-------------------")
    print(trainer_config)

    trainer.train()

    model_to_save = trainer.model.module if hasattr(
        # Take care of distributed/parallel training
        trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
