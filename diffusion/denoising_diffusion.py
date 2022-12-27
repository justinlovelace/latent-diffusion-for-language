import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator
import wandb

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.text_dataset as text_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation



ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
    ):
        super().__init__()

        self.diffusion_model = model
        if self.diffusion_model.class_conditional:
            if self.diffusion_model.class_unconditional_prob > 0:
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)


        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition

        self.max_seq_len = max_seq_len

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        register_buffer('latent_mean', torch.tensor([0]*self.latent_dim))
        register_buffer('latent_scale', torch.tensor(1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale+eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.latent_scale+eps)+self.latent_mean

    def diffusion_model_predictions(self, x, mask, t, x_self_cond = None, class_id=None):
        model_output = self.diffusion_model(x, mask, t, x_self_cond, class_id=class_id)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, class_id=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        latent = torch.randn(shape, device = device)
        mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, time_cond, self_cond, class_id=class_id)

            if time_next < 0:
                latent = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(latent)

            latent = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return (latent, mask)

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None):
        # TODO Create mask that controls length 
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        # TODO Implement for p_sample_loop 
        
        sample_fn = self.ddim_sample
        return sample_fn((batch_size, max_seq_len, latent_dim), length, class_id)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    #TODO handle masking 
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, mask, t, class_id, noise = None):
        b, l, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # Perform unconditional generation with some probability
        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes
        
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
            
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.diffusion_model_predictions(x, mask, t, class_id=class_id).pred_x_start.detach()

        # predict and take gradient step

        predictions = self.diffusion_model_predictions(x, mask, t, x_self_cond, class_id=class_id)        
                
        loss = self.loss_fn(predictions.pred_x_start, x_start, reduction = 'none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(x_start.shape[0])], 'b 1 -> b 1')

        return loss.mean()

    def forward(self, txt_latent, mask, class_id, *args, **kwargs):
        b, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(txt_latent, mask, t, class_id, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no',
        split_batches = True,
    ):
        super().__init__()


        set_seeds(42)

        self.args = args

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision,
            log_with='wandb'
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.accelerator.native_amp = amp

        self.diffusion = diffusion

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len

        # Init Encoder-decoder model
        assert 'bart' in args.enc_dec_model
        self.bart_model = BartForConditionalGeneration.from_pretrained(args.enc_dec_model)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.enc_dec_model)

        # dataset and dataloader
        dataset = text_dataset.get_dataset(
            dataset_name,
        )

        self.dataset = dataset.shuffle(seed=42)
        self.num_samples = min(self.num_samples,len(self.dataset['valid']['text']))
        # Subsample train and val splits for computing language generation during runtime

        self.dataloader = text_dataset.get_dataloader(args, dataset['train'], self.bart_model.config, self.tokenizer, self.max_seq_len)
        self.val_dataloader = text_dataset.get_dataloader(args, dataset['valid'], self.bart_model.config, self.tokenizer, self.max_seq_len)

        training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]
        length_counts = Counter(training_lengths)
        probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
        assert probs[0] == 0, 'Can\'t have examples of length 0'
        self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.diffusion.diffusion_model.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)
        
        # optimizer

        self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.bart_model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.diffusion, self.bart_model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # For backwards compatibility with earlier models
        self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def log_reference_metrics(self, test=False):
        accelerator = self.accelerator
        if test:
            train_subset = self.dataset['train']['text'][:self.num_samples]
            train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
            test_subset = self.dataset['test']['text'][:self.num_samples]
            self.reference_dict['reference/test_perplexity'] = evaluation.compute_perplexity(test_subset)
            for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
                self.reference_dict[f'reference/{mauve_model_id}_train_test_mauve'], _ = evaluation.compute_mauve(train_subset, test_subset, mauve_model_id)
                self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
                ngram_metrics = evaluation.compute_diversity(test_subset)
            for k, v in ngram_metrics.items():
                self.reference_dict[f"reference/test_{k}"] = v
            self.reference_dict[f"reference/test_memorization"] = evaluation.compute_memorization(test_subset, self.dataset['train']['text'])
            self.reference_dict['reference/test_unique_wordcount'] = evaluation.compute_wordcount(test_subset)
            return

        val_subset = self.dataset['valid']['text'][:self.num_samples]
        train_subset = self.dataset['train']['text'][:self.num_samples]
        train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
        self.reference_dict['reference/train_perplexity'] = evaluation.compute_perplexity(train_subset)
        self.reference_dict['reference/val_perplexity'] = evaluation.compute_perplexity(val_subset)
        for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_val_mauve'], _ = evaluation.compute_mauve(train_subset, val_subset, mauve_model_id)
            self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
        ngram_metrics = evaluation.compute_diversity(val_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/val_{k}"] = v
        ngram_metrics = evaluation.compute_diversity(train_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/train_{k}"] = v
        self.reference_dict[f"reference/val_memorization"] = evaluation.compute_memorization(val_subset, self.dataset['train']['text'])
        self.reference_dict['reference/train_unique_wordcount'] = evaluation.compute_wordcount(train_subset)
        self.reference_dict['reference/val_unique_wordcounts'] = evaluation.compute_wordcount(val_subset)
        torch.cuda.empty_cache() 


    @torch.no_grad()
    def gen_synthetic_dataset(self, num_samples, seed=42):
        num_classes = self.diffusion.diffusion_model.num_classes
        num_samples_per_class = num_samples//num_classes
        assert num_samples % num_classes == 0, f'Dataset size ({num_samples}) must be divisible by the number of classes ({num_classes})'
        data = {'text': [], 'label': []}
        self.ema.ema_model.eval()
        torch.manual_seed(seed)
        device = self.accelerator.device
        for class_id in range(num_classes):
            text = []
            while len(text) < num_samples_per_class:
                batches = num_to_groups(num_samples_per_class-len(text), self.eval_batch_size)
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=torch.tensor([class_id]*n, dtype=torch.long, device=device))), batches))
                
                for (latents, mask) in model_outputs:
                    latents, mask = latents.to(device), mask.to(device)
                    if self.args.normalize_latent:
                        latents = self.ema.ema_model.unnormalize_latent(latents)
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), **constant.generate_kwargs['beam'])
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    text.extend(texts_list)
            data['text'].extend(text)
            data['label'].extend([class_id]*num_samples_per_class)

        save_path = os.path.join(self.results_folder, f'synth_sample{num_samples}_seed{seed}.csv')
        print(save_path)
        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

            
            
    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.diffusion.to('cpu')
        torch.cuda.empty_cache() 

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = {}
        if exists(class_id):
            for filter_class_id in range(self.diffusion.diffusion_model.num_classes):
                filtered_dataset = self.dataset.filter(lambda example: example["label"]==filter_class_id)
                if test:
                    reference_texts[f'ref{filter_class_id}_test'] = filtered_dataset['test']['text']
                    continue
                reference_texts[f'ref{filter_class_id}_val'] = filtered_dataset['valid']['text']
                reference_texts[f'ref{filter_class_id}_train'] = filtered_dataset['train']['text']
            
            for key, reference_text in reference_texts.items():
                num_samples = min(num_samples, len(reference_text))
            reference_texts = {k: v[:num_samples] for k, v in reference_texts.items()}
        else:
            if test:
                reference_texts[f'test'] = self.dataset['test']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
            else:
                reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]

        milestone = self.step // self.save_and_sample_every
        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}    

        torch.manual_seed(seed)
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.diffusion.diffusion_model.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n))), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)
                for k, kwargs in constant.generate_kwargs.items():
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), **kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 

        metrics = {}

        self.ema.to('cpu')
        torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
            file_utils.save_text_samples(all_texts_list, os.path.join(self.results_folder, f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}{class_id_prefix}{strategy}-sample-{milestone}.txt'))
            metrics[f"model/{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"model/{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"model/{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/{class_id_prefix}samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable
            if metrics[f"model/{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        if len(self.reference_dict) == 0 or test:
            self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
        else:
            accelerator.log({**metrics,**self.reference_dict}, self.step)
        torch.cuda.empty_cache() 
        self.diffusion.to(device)
        self.ema.to(device)
        


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.


                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter).to(device)
                    with torch.no_grad():
                        latent = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask']).last_hidden_state
                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                latent_vecs = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
                                
                                # Add mean stats to model and EMA wrapper
                                self.diffusion.latent_mean = torch.mean(latent_vecs, dim=0)
                                self.ema.ema_model.latent_mean = self.diffusion.latent_mean

                                # Add var stats to model and EMA wrapper
                                self.diffusion.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)

                                self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                            latent = self.diffusion.normalize_latent(latent)

                    mask = data['attention_mask'].bool()
                    with self.accelerator.autocast():
                        loss = self.diffusion(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)


                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.diffusion.parameters())

                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                if not self.args.resume_training:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter).to(device)
                                latent = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask']).last_hidden_state
                                if self.args.normalize_latent or self.args.scale_latent:
                                    latent = self.diffusion.normalize_latent(latent)
                                    
                                mask = data['attention_mask'].bool()
                                with self.accelerator.autocast():
                                    loss = self.diffusion(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_loss += loss.item()
                                    loss = self.ema.ema_model(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                                    loss = loss / self.gradient_accumulate_every
                                    total_val_ema_loss += loss.item()


                            logs = {"loss": total_loss, "val_loss": total_val_loss, "val_ema_loss": total_val_ema_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), "samples": self.step*self.train_batch_size*self.gradient_accumulate_every}
                            pbar.set_postfix(**logs)
                            accelerator.log(logs, step=self.step)     
                        self.diffusion.train()           


                    if self.step % self.save_and_sample_every == 0:
                        self.sample()
                        if self.diffusion.diffusion_model.class_conditional:
                            for class_id in range(self.diffusion.diffusion_model.num_classes):
                                if self.args.dataset_name == 'ag_news':
                                    num_samples = 100
                                elif self.args.dataset_name == 'sst':
                                    num_samples = 500
                                self.sample(num_samples=num_samples, class_id=class_id)
                        self.save()
                        
                        self.diffusion.train() 

                pbar.update(1)

        accelerator.print('training complete')