"""Arguments for fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments as tArgs


@dataclass
class ModelArguments:
    """Model arguments for fine-tuning."""
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."}
    )


@dataclass
class DataArguments:
    """Data arguments for fine-tuning."""
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


@dataclass
class TrainingArguments(tArgs):
    """Training arguments for fine-tuning."""
    seed: int = field(default=42)
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": """Maximum sequence length. 
            Sequences will be right padded (and possibly truncated)."""
        },
    )
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = field(
        default_factory={"use_reentrant": False}.copy,
        metadata={
            "help": """Gradient checkpointing key word arguments such as `use_reentrant`. 
            Will be passed to `torch.utils.checkpoint.checkpoint` 
            through `model.gradient_checkpointing_enable`."""
        },
    )


@dataclass
class LoggingArguments:
    """Logging arguments for fine-tuning."""
    log_dir: str = field(
        default="logs",
        metadata={"help": "Directory to save logs."}
    )
    node_number: int = field(
        default=0,
        metadata={"help": "Node number."}
    )
