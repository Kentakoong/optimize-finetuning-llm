"""Main training script."""

import inspect
import logging
from llm_finetune.arguments import (DataArguments, LoggingArguments,
                                    ModelArguments, TrainingArguments)
from llm_finetune.dataset import make_supervised_data_module
from llm_finetune.trainer import EpochTimingCallback
from peft import LoraConfig
from torch import bfloat16, autocast, set_float32_matmul_precision
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser, set_seed)
from trl import SFTConfig, SFTTrainer

from mtnlog import JSONLogger, PerformanceLogger, PerformancePlotter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_dict(obj):
    """Extracts a dictionary from an object."""
    if hasattr(obj, '__dict__'):
        return extract_dict(obj.__dict__)

    return obj


def combine_keys_to_one_layer_dict(d, sep="."):
    """Combines keys to one layer dictionary."""
    new_dict = {}

    def recurse(inner_dict, parent_key=""):
        for key, value in inner_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                recurse(value, new_key)
            else:
                new_dict[new_key] = value

    recurse(d)
    return new_dict


def train():
    """Main training function."""

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoggingArguments))  # type: ignore

    parser_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, logging_args = parser_args  # type: ignore

    set_seed(training_args.seed)

    collector = PerformanceLogger(
        log_dir=f"{logging_args.log_dir}/metric",
        log_node=logging_args.node_number,
    )

    collector.change_tag("load_model")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bfloat16,
        bnb_4bit_quant_storage=bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        use_cache=False,
        torch_dtype=bfloat16,
        quantization_config=quantization_config,
        local_files_only=True
    )

    # model.to('cuda')

    model.config.attn_implementation = "flash_attention_2"


    collector.change_tag("load_token")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.eos_token

    collector.change_tag("load_data")

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    collector.change_tag("load_trainer")

    # Get the signature of SFTConfig.__init__
    sft_config_signature = inspect.signature(SFTConfig.__init__)
    sft_config_params = sft_config_signature.parameters

    # Filter out unexpected arguments
    filtered_args = {k: v for k, v in vars(training_args).items() if k in sft_config_params}

    # Explicitly pass all training arguments
    config = SFTConfig(**filtered_args)

    peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        **data_module,
        callbacks=[EpochTimingCallback()],
        peft_config=peft_config,
        packing=False,
    )

    all_dict = combine_keys_to_one_layer_dict(extract_dict({
        "model": model_args.__dict__,
        "data": data_args.__dict__,
        "training": training_args.__dict__,
        "logging": logging_args.__dict__,
        "quantization_config": quantization_config.__dict__,
        "model_config": model.config.__dict__,
        "tokenizer_config": tokenizer.__dict__,
        "trainer_config": trainer.args.__dict__,
    }))

    logger = JSONLogger(log_dir=logging_args.log_dir)

    logger.log(all_dict, filename="arguments")

    collector.change_tag("train")

    trainer.train()

    collector.change_tag("save_model")

    model_to_save = trainer.model.module if hasattr(
        # Take care of distributed/parallel training
        trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(training_args.output_dir)

    collector.stop()

    logger.log(trainer.state.log_history, filename="state")

    plotter = PerformancePlotter(base_dir=logging_args.log_dir, log_node=logging_args.node_number)

    plotter.plot()


if __name__ == "__main__":
    set_float32_matmul_precision('high')
    with autocast(device_type='cuda', dtype=bfloat16):
        train()