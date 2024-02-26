# Adapted from transformers pull request: https://github.com/huggingface/transformers/pull/18904
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase, T5ForConditionalGeneration


@dataclass
class DataCollatorForFlanLM:
    """
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: List[str]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        return batch
