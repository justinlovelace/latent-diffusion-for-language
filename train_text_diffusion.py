import argparse
from utils import file_utils
from transformers import AutoConfig
import json
import os
import numpy as np
import torch

import CONSTANTS
from diffusion.text_denoising_diffusion import GaussianDiffusion, Trainer
from model.diffusion_transformer import DiffusionTransformer

ATTN_HEAD_DIM=64

def get_diffusion_latent_dims(args):
    if args.latent_model_path is None:
        config = AutoConfig.from_pretrained(args.enc_dec_model)
        latent_dim = config.d_model
        lm_dim = config.d_model
    else:
        with open(os.path.join(args.latent_model_path, 'args.json'), 'rt') as f:
            latent_model_args = json.load(f)
        latent_dim = latent_model_args['dim_ae']
        lm_dim = 1024 if 'large' in latent_model_args['enc_dec_model'] else 768
    return latent_dim, lm_dim

def main(args):
    latent_dim, lm_dim = get_diffusion_latent_dims(args)
    # Override lm_dim if using different context LM from latent LM
    if 'large' in args.enc_dec_model:
        lm_dim = 1024
    elif 'xl' in args.enc_dec_model:
        lm_dim = 2048
    else:
        lm_dim = 768

    assert args.tx_dim%ATTN_HEAD_DIM==0, f'Transformer dimension must be divisible by {ATTN_HEAD_DIM}'
    model = DiffusionTransformer(
        tx_dim = args.tx_dim,
        tx_depth = args.tx_depth,
        heads = args.tx_dim//ATTN_HEAD_DIM,
        latent_dim = latent_dim,
        max_seq_len = args.max_seq_len,
        self_condition = args.self_condition,
        scale_shift = args.scale_shift,
        dropout = 0 if args.disable_dropout else 0.1,
        class_conditional= args.class_conditional,
        num_classes= (CONSTANTS.NUM_CLASSES[args.dataset_name] if args.class_conditional else 0),
        class_unconditional_prob= args.class_unconditional_prob,
        seq2seq=(args.dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
        seq2seq_context_dim=lm_dim, 
        num_dense_connections=args.num_dense_connections,
    ).cuda()

    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = model.max_seq_len,
        sampling_timesteps = args.sampling_timesteps,     # number of sampling steps
        sampler = args.sampler,
        train_schedule= args.train_schedule, 
        sampling_schedule= args.sampling_schedule,
        loss_type = args.loss_type,            # L1 or L2
        objective = args.objective,
        train_prob_self_cond = args.train_prob_self_cond,
        seq2seq_unconditional_prob = args.seq2seq_unconditional_prob,
        scale = args.scale,
    ).cuda()

    trainer = Trainer(
        args=args,
        diffusion=diffusion,
        dataset_name=args.dataset_name,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        gradient_accumulate_every = args.gradient_accumulation_steps,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        ema_update_every = args.ema_update_every,
        ema_decay = args.ema_decay,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        save_and_sample_every = args.save_and_sample_every,
        num_samples = args.num_samples,
        seq2seq_candidates = args.seq2seq_candidates,
        results_folder = args.output_dir,
        amp = args.amp,
        mixed_precision = args.mixed_precision,
    )

    if args.eval:
        trainer.load(args.resume_dir, best=trainer.diffusion.diffusion_model.seq2seq)
        if trainer.diffusion.diffusion_model.seq2seq:
            trainer.sample_seq2seq(cls_free_guidance=2.0, incremental=False)
        else:
            trainer.sample()
        if args.class_conditional:
            for class_id in range(model.num_classes):
                trainer.sample(class_id=class_id)
        return
    if args.eval_test:
        trainer.load(args.resume_dir, best=trainer.diffusion.diffusion_model.seq2seq)
        if trainer.diffusion.diffusion_model.seq2seq:
            # trainer.sample_seq2seq(split='test', incremental=False)
            trainer.sample_seq2seq(split='test', cls_free_guidance=2.0, incremental=False)
        else:
            for seed in [42, 43, 44, 45, 46]:
                trainer.dataset = trainer.dataset.shuffle(seed)
                trainer.sample(seed=seed, test=True)
                if args.class_conditional:
                    for class_id in range(model.num_classes):
                        trainer.sample(class_id=class_id, seed=seed, test=True)
        return

    if args.resume_training:
        trainer.load(args.resume_dir)
    if args.init_path:
        trainer.load(args.init_path, init_only=True)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_diff_models")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    # Optimization hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=60000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_every", type=int, default=1)
    # Diffusion Hyperparameters
    parser.add_argument(
        "--objective",
        type=str,
        default="pred_noise",
        choices=["pred_noise", "pred_x0", "pred_v",],
        help=(
            "Which parameterization to use for the diffusion objective."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "smooth_l1"],
        help=(
            "Which loss function to use for diffusion."
        ),
    )
    parser.add_argument(
        "--train_schedule",
        type=str,
        default="cosine",
        choices=["beta_linear", "simple_linear", "cosine", 'sigmoid'],
        help=(
            "Which noise schedule to use."
        ),
    )
    parser.add_argument(
        "--sampling_schedule",
        type=str,
        default=None,
        choices=["beta_linear", "cosine", "simple_linear", None],
        help=(
            "Which noise schedule to use."
        ),
    )

    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sampling_timesteps", type=int, default=250)
    parser.add_argument("--normalize_latent", action="store_true", default=False)
    # Generation Arguments
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seq2seq_candidates", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--train_prob_self_cond", type=float, default=0.5)
    parser.add_argument(
        "--sampler",
        type=str,
        default='ddpm',
        choices=["ddpm", "ddim", "dpmpp"],
        help=(
            "Which noise schedule to use."
        ),
    )
    # Model hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--tx_dim", type=int, default=512)
    parser.add_argument("--tx_depth", type=int, default=6)
    parser.add_argument("--scale_shift", action="store_true", default=False)
    parser.add_argument("--num_dense_connections", type=int, default=0)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--class_conditional", action="store_true", default=False)
    parser.add_argument("--class_unconditional_prob", type=float, default=.1)
    parser.add_argument("--seq2seq_unconditional_prob", type=float, default=0.1)
    # Accelerate arguments
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--latent_model_path", type=str, default=None)
    parser.add_argument("--init_path", type=str, default=None)
    
    args = parser.parse_args()
    assert not (args.eval and args.resume_training)
    if args.eval or args.resume_training:
        assert args.resume_dir is not None

    if args.eval or args.resume_training or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        # Hold out sampling/evaluation parameters
        heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'num_samples', 'sampling_timesteps', 'sampling_schedule', 'seq2seq_candidates', 'scale', 'sampler', 'resume_training'}
        # Overwrite args with saved args
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v
    main(args)
