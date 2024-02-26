import torch
import torch.nn as nn

from dataclasses import dataclass

from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration

from latent_models.perceiver_ae import PerceiverAutoEncoder
from einops import rearrange


class T5ForConditionalGenerationLatent(T5ForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
            
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        
        return encoder_outputs
    

class MT5ForConditionalGenerationLatent(MT5ForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents

        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents, max_seq_len=192)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
            
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        
        return encoder_outputs