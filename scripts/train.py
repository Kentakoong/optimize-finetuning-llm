import inspect
import time
from dataclasses import asdict

from llm_finetune.arguments import (DataArguments, ModelArguments,
                                    TrainingArguments)
from llm_finetune.dataset import make_supervised_data_module
from llm_finetune.trainer import EpochTimingCallback
from peft import LoraConfig
from torch import bfloat16
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser, set_seed)
from trl import SFTConfig, SFTTrainer


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # Set the quantization type to "nf4", which stands for "near-float 4-bit". This is a quantization scheme designed to maintain high accuracy with lower bit rates.
        bnb_4bit_quant_type="nf4",
        # Specify the data type for computation. Here it uses 16-bit floating points as defined above.
        bnb_4bit_compute_dtype=bfloat16,
        bnb_4bit_quant_storage=bfloat16,
        # Determines whether to use double quantization. Setting this to False uses single quantization, which is simpler and faster.
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=bfloat16,
        use_cache=False,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2"
    )

    print("------ Memory Footprint of the model ------")
    print(model.get_memory_footprint())

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

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
        ** data_module,
        callbacks=[EpochTimingCallback()],
        peft_config=peft_config,
        packing=False,
    )

    trainer.train()

    model_to_save = trainer.model.module if hasattr(
        # Take care of distributed/parallel training
        trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
