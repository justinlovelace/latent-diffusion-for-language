import torch
from evaluate import load
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
from nltk.util import ngrams
from collections import defaultdict
import spacy
import numpy as np
import wandb

def compute_perplexity(all_texts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device='cuda')
    return results['mean_perplexity']

def compute_wordcount(all_texts_list):
    wordcount = load("word_count")
    wordcount = wordcount.compute(data=all_texts_list)
    return wordcount['unique_words']

def compute_diversity(all_texts_list):
    ngram_range = [2,3,4]

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    token_list = []
    for sentence in all_texts_list:
        token_list.append([str(token) for token in tokenizer(sentence)])
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n])
    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity
    return metrics

def compute_memorization(all_texts_list, human_references, n=4):

    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    return duplicate/total

def compute_mauve(all_texts_list, human_references, model_id):
    torch.cuda.empty_cache() 
    assert model_id == 'gpt2-large'
    mauve = load("mauve")

    results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name=model_id, max_text_length=256, device_id=0)

    assert len(all_texts_list) == len(human_references)
    
    return results.mauve, results.divergence_curve

def compute_bleu(all_texts_list, human_references):
    bleu = load("bleu")

    human_references = [[ref] for ref in human_references]
    results = bleu.compute(predictions=all_texts_list, references=human_references)
    
    return results['bleu']

def compute_sacrebleu(all_texts_list, human_references, tokenize, use_effective_order=False):
    sacrebleu = load("sacrebleu")

    human_references = [[ref] for ref in human_references]
    results = sacrebleu.compute(predictions=all_texts_list, references=human_references, tokenize=tokenize, use_effective_order=use_effective_order)
    
    return results['score']

def compute_debertascore(all_texts_list, human_references):
    bert = load("bertscore")

    human_references = [[ref] for ref in human_references]
    results = bert.compute(predictions=all_texts_list, references=human_references, lang="en", model_type='microsoft/deberta-xlarge-mnli')

    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()
    
    return results
    

def compute_bertscore(all_texts_list, human_references):
    bert = load("bertscore")

    human_references = [[ref] for ref in human_references]
    results = bert.compute(predictions=all_texts_list, references=human_references, lang="en", rescale_with_baseline=True)

    del results['hashcode']
    for key, value in results.items():
        results[key] = np.asarray(value).mean()
    
    return results

def compute_rouge(all_texts_list, human_references, use_aggregator=True, use_stemmer=False):
    rouge = load("rouge")

    human_references = [[ref] for ref in human_references]
    results = rouge.compute(predictions=all_texts_list, references=human_references, use_aggregator=use_aggregator, use_stemmer=use_stemmer)
    
    return results

