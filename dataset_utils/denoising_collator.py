# Adapted from transformers pull request: https://github.com/huggingface/transformers/pull/18904
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()

        return batch



def main():
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    from text_dataset import get_dataset
    dataset = get_dataset('e2e')
    def tokenization(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)
    dataset = dataset.map(tokenization, remove_columns='text')
    # import pdb; pdb.set_trace()
    
    dl = DataLoader(
        dataset['train'],
        collate_fn=DataCollatorForBartDenoisingLM(tokenizer),
        batch_size=4,
        shuffle=True,
    )
    for b in dl:
        print(f'label: {tokenizer.batch_decode(b["labels"])}')
        print(f'input_ids: {tokenizer.batch_decode(b["input_ids"])}')   
        generated_ids = model.generate(b["input_ids"], attention_mask=b["attention_mask"], max_length=64)
        print(f'output: {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}')   


if __name__ == "__main__":
    main()