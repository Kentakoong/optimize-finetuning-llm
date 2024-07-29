import transformers
from torch import Tensor

from torch.nn.utils.rnn import pad_sequence

from typing import Sequence, Dict
from dataclasses import dataclass

from .constant import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")  # noqa: E501
        )
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
