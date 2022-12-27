from multiprocessing.spawn import prepare
import os

from datasets import load_dataset, Value
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator

from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM

def exists(x):
    return x is not None

def get_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'e2e':
        e2e_data_path = 'datasets/e2e_data'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(e2e_data_path, f'src1_{split}.txt') for split in ['train', 'valid', 'test']})
        dataset = process_e2e_dataset(dataset, metadata=metadata)
    elif dataset_name == 'roc':
        roc_data_path = 'datasets/ROCstory'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc_dataset(dataset)
    elif dataset_name == 'sst':
        dataset = load_dataset("sst")
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_sst_dataset(dataset)
    elif dataset_name == 'ag_news':
        dataset = load_dataset('pietrolesci/ag_news', 'original')
        train_ds = dataset['train']
        train_val_ds = train_ds.train_test_split(test_size=1000, seed=42)
        train_val_ds['valid'] = train_val_ds['test']
        train_val_ds['test'] = dataset['test']
        dataset = process_ag_news_dataset(train_val_ds)
    else:
        raise NotImplementedError
    if exists(synthetic_train_path):
        synth_train_data = load_dataset("csv", data_files={f'train': synthetic_train_path})
        dataset['train'] = synth_train_data['train']
    return dataset

def process_e2e_dataset(dataset, metadata=False):
    def extract_e2e_text(example):
        split_text = example['text'].split('||')
        assert len(split_text) == 2
        assert not split_text[1].isspace()
        parsed = {'text': PreTrainedTokenizerBase.clean_up_tokenization(split_text[1].strip())}
        if metadata:
            meta_strings = split_text[0].split(' | ')
            parsed['label'] = meta_strings
        return parsed
    dataset = dataset.map(extract_e2e_text, load_from_cache_file=False)
    return dataset

def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, load_from_cache_file=False)
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_sst_dataset(dataset):
    def process_sst_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example['sentence'].strip()), 'label':0 if example['label'] < 0.5 else 1}
    dataset = dataset.map(process_sst_text, load_from_cache_file=False, remove_columns=['sentence', 'tokens', 'tree']).cast_column('label', Value(dtype='int64', id=None))
    return dataset

def process_ag_news_dataset(dataset):
    def process_ag_news_text(example):
        # return {'text': PreTrainedTokenizerBase.clean_up_tokenization(f'Title: {example["title"]}<pad> Description: {example["description"]}'.strip()), 'label':example['label']-1}
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["description"].strip()), 'label':example['label']-1}
    dataset = dataset.map(process_ag_news_text, load_from_cache_file=False, remove_columns=['title', 'description', 'class'])
    return dataset


def get_dataloader(args, dataset, model_config, tokenizer, max_seq_len, mode='diffusion'):
    assert mode in {'diffusion', 'classification', 'language_modeling'}
    def tokenization(example):
        if mode == 'language_modeling':
            text = tokenizer.bos_token + example["text"] + tokenizer.eos_token
            tokenized_text = tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors='pt')
            tokenized_text['labels'] = tokenized_text['input_ids'].clone()
            tokenized_text['labels'][tokenized_text['labels'] == tokenizer.pad_token_id] = -100
            return tokenized_text
        else:
            text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

    if mode=='classification':
        dataset = dataset.map(tokenization, remove_columns='text')
        return DataLoader(
                dataset,
                collate_fn=default_data_collator,
                batch_size=args.train_batch_size,
                shuffle=True,
                pin_memory = True
            )
    if mode == 'language_modeling':
        if "label" in dataset.column_names:
            dataset = dataset.remove_columns("label")
        dataset = dataset.map(tokenization, remove_columns='text')
        return DataLoader(
                dataset,
                collate_fn=default_data_collator,
                batch_size=args.train_batch_size,
                shuffle=True,
                pin_memory = True
            )
    
    dataset = dataset.map(tokenization, remove_columns='text')
    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory = True
        )
    return dl

if __name__ == "__main__":

    dataset = get_dataset('roc')
    # import pdb; pdb.set_trace()
    print(dataset['train'][0])