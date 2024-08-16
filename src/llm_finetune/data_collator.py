"""Data collators for fine-tuning language models."""

from dataclasses import dataclass
from typing import Dict, Sequence

from transformers import PreTrainedTokenizer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .constant import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")  # noqa: E501
        )
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=float(self.tokenizer.pad_token_id or 0),
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(float(self.tokenizer.pad_token_id or 0)),
        )
