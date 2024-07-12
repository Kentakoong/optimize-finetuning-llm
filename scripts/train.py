from transformers import Trainer, set_seed, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
import time

from llm_finetune.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from llm_finetune.dataset import make_supervised_data_module
from llm_finetune.trainer import EpochTimingCallback


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config
    )

    print("------ Memory Footprint of the model ------")
    print(model.get_memory_footprint())

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module, callbacks=[EpochTimingCallback()]
    )

    start_time = time.time()

    trainer.train(training_args.checkpoint)

    print(f"--- execute time : {time.time() - start_time} seconds ---")

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
