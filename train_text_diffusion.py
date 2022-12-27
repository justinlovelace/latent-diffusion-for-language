import argparse
from utils import file_utils
from transformers import AutoConfig
import json
import os
import numpy as np

import CONSTANTS
from diffusion.denoising_diffusion import GaussianDiffusion, Trainer
from model.diffusion_transformer import DiffusionTransformer

ATTN_HEAD_DIM=64

def main(args):
    config = AutoConfig.from_pretrained(args.enc_dec_model)
    assert args.tx_dim%ATTN_HEAD_DIM==0, f'Transformer dimension must be divisible by {ATTN_HEAD_DIM}'
    model = DiffusionTransformer(
        tx_dim = args.tx_dim,
        tx_depth = args.tx_depth,
        heads = args.tx_dim//ATTN_HEAD_DIM,
        latent_dim = config.d_model,
        self_condition = args.self_condition,
        scale_shift = args.scale_shift,
        dropout = 0 if args.disable_dropout else 0.1,
        class_conditional = args.class_conditional,
        num_classes = (CONSTANTS.NUM_CLASSES[args.dataset_name] if args.class_conditional else 0),
        class_unconditional_prob= args.class_unconditional_prob,
    ).cuda()

    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = model.max_seq_len,
        timesteps = args.timesteps,           # number of steps
        sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = args.loss_type,            # L1 or L2
        beta_schedule = args.beta_schedule,
        p2_loss_weight_gamma = args.p2_loss_weight_gamma,
        objective = args.objective,
        ddim_sampling_eta=args.ddim_sampling_eta,
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
        results_folder = args.output_dir,
        amp = args.amp,
        mixed_precision = args.mixed_precision,
    )

    if args.gen_data:
        trainer.load(args.resume_dir)
        for seed in [42, 43, 44, 45, 46]:
            trainer.gen_synthetic_dataset(num_samples=args.num_samples, seed=seed)
        return
    if args.eval:
        trainer.load(args.resume_dir)
        trainer.sample()
        if args.class_conditional:
            for class_id in range(model.num_classes):
                trainer.sample(class_id=class_id)
        return
    if args.eval_test:
        trainer.load(args.resume_dir)
        
        for seed in [42, 43, 44, 45, 46]:
            trainer.dataset = trainer.dataset.shuffle(seed)
            trainer.sample(seed=seed, test=True)
            if args.class_conditional:
                for class_id in range(model.num_classes):
                    trainer.sample(class_id=class_id, seed=seed, test=True)
        return
    if args.resume_training:
        trainer.load(args.resume_dir)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    # Data Hyperparameters
    parser.add_argument("--corruption_prob", type=float, default=.0)
    # Optimization hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=60000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
        choices=["pred_noise", "pred_x0"],
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
        "--beta_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help=(
            "Which noise schedule to use."
        ),
    )
    parser.add_argument("--p2_loss_weight_gamma", type=float, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=250)
    parser.add_argument("--normalize_latent", action="store_true", default=False)
    # Generation Arguments
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--ddim_sampling_eta", type=float, default=1)
    # Model hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--tx_dim", type=int, default=512)
    parser.add_argument("--tx_depth", type=int, default=6)
    parser.add_argument("--scale_shift", action="store_true", default=False)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--class_conditional", action="store_true", default=False)
    parser.add_argument("--class_unconditional_prob", type=float, default=.1)
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
    parser.add_argument("--gen_data", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--milestone", type=int, default=12)
    
    args = parser.parse_args()
    assert not (args.eval and args.resume_training)
    if args.eval or args.resume_training:
        assert args.resume_dir is not None

    if args.eval or args.resume_training or args.gen_data or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        if args.eval or args.gen_data or args.eval_test:
            heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'gen_data', 'ddim_sampling_eta', 'num_samples', 'sampling_timesteps'}
        else:
            heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'resume_training', 'ddim_sampling_eta', 'num_samples', 'sampling_timesteps', 'save_and_sample_every'}
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v

    if args.output_dir is None:
        args.output_dir = file_utils.get_output_dir(args)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    main(args)
