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
    assert model_id in {'gpt2-large', 'all-mpnet-base-v2'}
    mauve = load("mauve")
    assert len(all_texts_list) == len(human_references)

    if model_id == 'all-mpnet-base-v2':
        model = SentenceTransformer(model_id).cuda()
        #Sentences are encoded by calling model.encode()
        all_texts_list_embedding = model.encode(all_texts_list)
        human_references_embedding = model.encode(human_references)
        results = mauve.compute(predictions=all_texts_list, p_features=all_texts_list_embedding, references=human_references, q_features=human_references_embedding, featurize_model_name=model_id, max_text_length=256, device_id=0, mauve_scaling_factor=8,)
    elif model_id == 'gpt2-large':
        results = mauve.compute(predictions=all_texts_list, references=human_references, featurize_model_name=model_id, max_text_length=256, device_id=0)
    else:
        raise NotImplementedError
    
    return results.mauve, results.divergence_curve

    