from multiprocessing.spawn import prepare
import os
import json

from datasets import load_dataset, Value
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator

from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM
from dataset_utils.flan_collator import DataCollatorForFlanLM

def exists(x):
    return x is not None


def get_dataset(dataset_name, metadata=False, synthetic_train_path=None):
    if dataset_name == 'roc':
        roc_data_path = 'datasets/ROCstory'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc_dataset(dataset)
    elif dataset_name == 'ag_news':
        dataset = load_dataset('pietrolesci/ag_news', 'original')
        train_ds = dataset['train']
        train_val_ds = train_ds.train_test_split(test_size=1000, seed=42)
        train_val_ds['valid'] = train_val_ds['test']
        train_val_ds['test'] = dataset['test']
        dataset = process_ag_news_dataset(train_val_ds)
    elif dataset_name == 'xsum':
        dataset = load_dataset('xsum')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_xsum_dataset(dataset)
    elif dataset_name == 'qqp':
        qqp_data_path = 'datasets/qqp'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(qqp_data_path, f'{split}.jsonl') for split in ['train', 'valid', 'test']})
        dataset = process_qqp_dataset(dataset)
    elif dataset_name == 'wmt14-de-en':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-en')
    elif dataset_name == 'wmt14-de-de':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'de-de')
    elif dataset_name == 'wmt14-en-de':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-de')
    elif dataset_name == 'wmt14-en-en':
        dataset = load_dataset('wmt14', 'de-en')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_wmt14_dataset(dataset, 'en-en')
    else:
        raise NotImplementedError
    return dataset


def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_ag_news_dataset(dataset):
    def process_ag_news_text(example):
        # return {'text': PreTrainedTokenizerBase.clean_up_tokenization(f'Title: {example["title"]}<pad> Description: {example["description"]}'.strip()), 'label':example['label']-1}
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["description"].strip()), 'label':example['label']-1}
    dataset = dataset.map(process_ag_news_text, remove_columns=['title', 'description', 'class'])
    return dataset

def process_xsum_dataset(dataset):
    def process_xsum_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["summary"].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example["document"].strip())}
    dataset = dataset.map(process_xsum_text, remove_columns=['summary', 'document', 'id'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_qqp_dataset(dataset):
    def process_qqp_text(example):
        dict_example = json.loads(example['text'])
        dict_example['text'] = dict_example['trg']
        dict_example['context'] = dict_example['src']
        del dict_example['trg']
        del dict_example['src']
        return dict_example
    dataset = dataset.map(process_qqp_text, )
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_wmt14_dataset(dataset, lang_pair):
    def process_wmt14_text(example, lang_pair):
        source, target = lang_pair.split('-')
        assert source in ['de', 'en']
        assert target in ['de', 'en']

        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][target].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example['translation'][source].strip())}
    dataset = dataset.map(process_wmt14_text, fn_kwargs={'lang_pair': lang_pair}, remove_columns=['translation'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def parse_metadata(metadata):
    if type(metadata) == list:
        return ' | '.join(metadata)
    elif type(metadata) == float:
        return 'Positive' if metadata > 0.5 else 'Negative'


def get_dataloader(args, dataset, model_config, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        # print('EXAMPLE: ', example)
        if mode == 'diffusion' and args.dataset_name in {'xsum', 'qqp',  'wmt14-en-de', 'wmt14-de-en'}:
            # import pdb; pdb.set_trace()
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']

            if args.dataset_name in {'qqp', 'wmt14-en-de', 'wmt14-de-en'}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            elif args.dataset_name in {'xsum',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len*4)
            else:
                raise NotImplementedError

            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)
            
            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]

            return model_inputs
        else:
            text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

    if 'mbart' in args.enc_dec_model:
        collate_fn=default_data_collator
    elif 'bart' in args.enc_dec_model:
        collate_fn=DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM(tokenizer)
    else:
        raise NotImplementedError
    
    if args.dataset_name in {'xsum', 'qqp'} or 'wmt14' in args.dataset_name:
        dataset = dataset.map(tokenization, remove_columns=['text', 'context'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns='text')
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl

if __name__ == "__main__":

    dataset = get_dataset('roc')
    print(dataset['train'][0])