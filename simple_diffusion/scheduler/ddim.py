# Disclaimer: This code was influenced by
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import math
import numpy as np
import torch
from tqdm import tqdm

from simple_diffusion.utils import unnormalize_to_zero_to_one


def cosine_beta_schedule(timesteps, beta_start=0.0, beta_end=0.999, s=0.008):
    """Standard cosine beta schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


def shifted_cosine_beta_schedule(timesteps, shift=1.0, beta_start=0.0, beta_end=0.999, s=0.008):
    """Shifted cosine schedule - better for high-res and detailed images.
    
    Args:
        timesteps: Number of diffusion steps
        shift: Schedule shift factor. >1 pushes more signal to later timesteps,
               which can help with learning fine details.
        beta_start: Minimum beta value
        beta_end: Maximum beta value
        s: Small offset to prevent singularity
    
    Returns:
        Beta schedule tensor
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Apply shift - pushes more signal retention to later timesteps
    if shift != 1.0:
        alphas_cumprod = alphas_cumprod ** shift
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


def enforce_zero_terminal_snr(alphas_cumprod):
    """Enforce zero terminal SNR for better training dynamics.
    
    This ensures the final timestep has approximately zero signal,
    preventing the model from "cheating" by relying on residual signal.
    """
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    
    alphas_cumprod_sqrt = alphas_cumprod_sqrt - alphas_cumprod_sqrt_T
    alphas_cumprod_sqrt = alphas_cumprod_sqrt * alphas_cumprod_sqrt_0 / (
        alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T
    )
    
    return alphas_cumprod_sqrt ** 2


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
        beta_schedule_shift=1.0,
        prediction_type="epsilon",
        clip_sample=True,
        set_alpha_to_one=True,
        rescale_betas_zero_snr=False,
    ):
        """Initialize the DDIM scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion steps
            beta_start: Starting beta value for linear schedule
            beta_end: Ending beta value for linear schedule
            beta_schedule: Schedule type - "cosine", "shifted_cosine", or "linear"
            beta_schedule_shift: Shift factor for shifted_cosine (>1 = more noise)
            prediction_type: "epsilon" for noise prediction, "v_prediction" for velocity
            clip_sample: Whether to clip predicted x0 to [-1, 1]
            set_alpha_to_one: Whether to set final alpha to 1
            rescale_betas_zero_snr: Whether to enforce zero terminal SNR
        """
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # Build beta schedule
        if beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
        elif beta_schedule == "shifted_cosine":
            self.betas = shifted_cosine_beta_schedule(
                num_train_timesteps, shift=beta_schedule_shift, 
                beta_start=beta_start, beta_end=beta_end
            )
        else:  # linear
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Optionally enforce zero terminal SNR
        if rescale_betas_zero_snr:
            self.alphas_cumprod = enforce_zero_terminal_snr(self.alphas_cumprod)
        
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.timesteps = np.arange(num_train_timesteps)[::-1]

    def _set_timesteps(self, num_inference_steps, offset=0):
        self.timesteps = (
            np.arange(
                0,
                self.num_train_timesteps,
                self.num_train_timesteps // num_inference_steps,
            )[::-1] + offset
        )
    
    def get_velocity(self, sample, noise, timesteps):
        """Compute velocity for v-prediction training.
        
        v = sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * sample
        
        This parameterization is more stable and often produces better results,
        especially for high-resolution images.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(sample.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None, None]
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        return (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

    def _step(self, model_output, timestep, sample, eta=1.0, generator=None):
        """Perform one denoising step.
        
        Args:
            model_output: The model's prediction (noise, velocity, or x0 depending on prediction_type)
            timestep: Current timestep
            sample: Current noisy sample
            eta: Stochasticity parameter (0 = DDIM, 1 = DDPM)
            generator: Random generator for reproducibility
        
        Returns:
            Denoised sample at previous timestep
        """
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // len(self.timesteps)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. Compute predicted original sample from model output
        if self.prediction_type == "v_prediction":
            # v-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * x0
            # Solving for x0: x0 = sqrt(alpha) * sample - sqrt(1-alpha) * v
            pred_original_sample = (
                alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            )
            # Derive noise from v-prediction for the direction step
            pred_noise = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            # epsilon prediction (standard)
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
            pred_noise = model_output
    
        # 4. Clamp "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
            # Re-derive noise from clipped x0 (Glide approach)
            pred_noise = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_noise

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator).to(sample.device)
            prev_sample += std_dev_t * noise

        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(timesteps.device)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        return (
            sqrt_alpha_prod[:, None, None, None] * original_samples
            + sqrt_one_minus_alpha_prod[:, None, None, None] * noise
        )

    @torch.no_grad()
    def generate(
        self,
        model,
        batch_size=1,
        generator=None,
        eta=1.0,
        num_inference_steps=50,
        device=None,
        # === CFG parameters (for future conditional generation) ===
        context=None,
        guidance_scale=1.0,
        unconditional_context=None,
    ):
        """Generate images using DDIM sampling.
        
        Args:
            model: The denoising model
            batch_size: Number of images to generate
            generator: Random generator for reproducibility
            eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM-like)
            num_inference_steps: Number of denoising steps
            device: Device to run on
            context: Conditioning embeddings (for future text-to-image)
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
            unconditional_context: Null/empty conditioning for CFG
        
        Returns:
            Dictionary with generated samples
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.randn(
            (batch_size, model.in_channels, model.sample_size, model.sample_size),
            generator=generator,
        ).to(device)
        self._set_timesteps(num_inference_steps)
        
        # Move alphas to device for v-prediction support
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        # CFG: Check if we need classifier-free guidance
        do_cfg = guidance_scale > 1.0 and context is not None

        for t in tqdm(self.timesteps):
            if do_cfg:
                # Classifier-Free Guidance: run model twice
                # Once with conditioning, once without
                model_output_cond = model(image, t, context=context)["sample"]
                model_output_uncond = model(image, t, context=unconditional_context)["sample"]
                
                # CFG formula: output = uncond + scale * (cond - uncond)
                model_output = model_output_uncond + guidance_scale * (
                    model_output_cond - model_output_uncond
                )
            else:
                # Unconditional or no CFG
                model_output = model(image, t)["sample"]
            
            # Predict previous mean of image x_t-1 and add variance depending on eta
            # do x_t -> x_t-1
            image = self._step(model_output, t, image, eta, generator=generator)
        
        image = unnormalize_to_zero_to_one(image)
        return {"sample": image.cpu().permute(0, 2, 3, 1).numpy(), "sample_pt": image}

    def __len__(self):
        return self.num_train_timesteps
