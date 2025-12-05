import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet
from simple_diffusion.utils import save_images
from simple_diffusion.dataset import CustomDataset, get_dataset
import pandas as pd
import webdataset as wds

from simple_diffusion.ema import EMA

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

n_timesteps = 1000
n_inference_timesteps = 250

def _grayscale_to_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def sample_timesteps_lognormal(batch_size, num_timesteps, mean=0.0, std=1.0, device='cpu'):
    """Sample timesteps using log-normal distribution.
    
    This biases sampling towards middle timesteps where most learning happens,
    rather than uniform sampling which overweights very noisy/clean timesteps.
    """
    u = torch.normal(mean=mean, std=std, size=(batch_size,), device=device)
    u = torch.sigmoid(u)  # Map to [0, 1]
    timesteps = (u * num_timesteps).long().clamp(0, num_timesteps - 1)
    return timesteps


def get_snr_weights(timesteps, alphas_cumprod, gamma=5.0):
    """Compute Min-SNR-γ weights for loss rebalancing.
    
    From: https://arxiv.org/abs/2303.09556
    This prevents high-SNR (low noise) timesteps from dominating the loss.
    
    Args:
        timesteps: Sampled timesteps
        alphas_cumprod: Cumulative product of alphas from noise schedule
        gamma: SNR clipping value (default 5.0 works well)
    
    Returns:
        Per-sample weights for the loss
    """
    alphas = alphas_cumprod[timesteps]
    snr = alphas / (1 - alphas)
    # Clamp SNR and compute weight
    weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return weight


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model initialization - conditioning placeholder for future CFG support
    # When adding text conditioning, add: context_dim=768 (or your text encoder dim)
    model = UNet(3, image_size=args.resolution, hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=args.use_flash_attn)
    
    # Scheduler with v-prediction and shifted cosine schedule
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=n_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        beta_schedule_shift=args.beta_schedule_shift,
    )
    model = model.to(device)

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    tfms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor(),
        # normalize to [-1, 1] for faster convergence and numerical stability
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    if args.dataset_name in ["combined", "yfcc7m"]:
        dataset = get_dataset(args.dataset_name,
                              args.dataset_path,
                              transforms=tfms)
    elif args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )

        def aug(examples):
            images = [
                tfms(image) for image in examples["image"]
            ]
            return {"image": images}

        dataset.set_transform(aug)
    else:
        def aug(examples):
            images = [
                tfms(image) for image in examples
            ]
            return {"image": images}

        df = pd.read_pickle(args.dataset_path)
        dataset = CustomDataset(df, tfms)

    if args.dataset_name == "yfcc7m":
        train_dataloader = wds.WebLoader(dataset,
                                         num_workers=2,
                                         batch_size=args.train_batch_size)
        # hardcoded for num images = 7329280 :/
        steps_per_epcoch = 7329280 // args.train_batch_size
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True)
        steps_per_epcoch = len(train_dataloader)

    total_num_steps = (steps_per_epcoch * args.num_epochs) // args.gradient_accumulation_steps
    total_num_steps += int(total_num_steps * 10/100)
    gamma = args.gamma
    ema = EMA(model, gamma, total_num_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_num_steps,
    )

    summary(model, [(1, 3, args.resolution, args.resolution), (1,)], verbose=1)

    scaler = GradScaler(enabled=args.fp16_precision)
    global_step = 0
    losses = []
    
    # Get alphas_cumprod for min-SNR weighting
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=steps_per_epcoch)
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            orig_images = batch["image"].to(device)
            batch_size = orig_images.shape[0]
            
            # Generate noise with optional noise offset for better color distribution
            noise = torch.randn(orig_images.shape, device=device)
            if args.noise_offset > 0:
                # Add channel-wise noise offset - helps with very dark/bright images
                noise = noise + args.noise_offset * torch.randn(
                    batch_size, orig_images.shape[1], 1, 1, device=device
                )
            
            # Sample timesteps using log-normal distribution (or uniform if disabled)
            if args.use_lognormal_sampling:
                timesteps = sample_timesteps_lognormal(
                    batch_size, 
                    noise_scheduler.num_train_timesteps,
                    mean=args.lognormal_mean,
                    std=args.lognormal_std,
                    device=device
                )
            else:
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (batch_size,), device=device
                ).long()
            
            noisy_images = noise_scheduler.add_noise(orig_images, noise, timesteps)

            optimizer.zero_grad()
            with autocast(enabled=args.fp16_precision):
                # Model prediction - currently unconditional
                # For CFG: pass context=text_embeddings here, and randomly drop context
                # during training (e.g., 10% of the time set context=None)
                model_output = model(noisy_images, timesteps)["sample"]
                
                # Compute target based on prediction type
                if args.prediction_type == "v_prediction":
                    # v-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * sample
                    target = noise_scheduler.get_velocity(orig_images, noise, timesteps)
                else:
                    # epsilon prediction (standard)
                    target = noise
                
                # Compute per-sample loss
                if args.loss_type == "l1":
                    loss_per_sample = F.l1_loss(model_output, target, reduction='none')
                else:  # l2/mse
                    loss_per_sample = F.mse_loss(model_output, target, reduction='none')
                
                # Reduce over non-batch dimensions
                loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])
                
                # Apply Min-SNR-γ weighting
                if args.use_snr_weighting:
                    snr_weights = get_snr_weights(
                        timesteps, alphas_cumprod, gamma=args.snr_gamma
                    )
                    loss = (loss_per_sample * snr_weights).mean()
                else:
                    loss = loss_per_sample.mean()

            scaler.scale(loss).backward()
            
            # Fix: Gradient clipping BEFORE optimizer step
            if args.use_clip_grad:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            ema.update_params(gamma)
            gamma = ema.update_gamma(global_step)

            lr_scheduler.step()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "gamma": gamma
            }

            progress_bar.set_postfix(**logs)
            global_step += 1

            # Generate sample images for visual inspection
            if global_step % args.save_model_steps == 0:
                ema.ema_model.eval()
                with torch.no_grad():
                    # has to be instantiated every time, because of reproducibility
                    generator = torch.manual_seed(0)
                    generated_images = noise_scheduler.generate(
                        ema.ema_model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator,
                        eta=1.0,
                        batch_size=args.eval_batch_size)

                    save_images(generated_images, epoch, args)

                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            'ema_model_state': ema.ema_model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                        }, args.output_dir)

        progress_bar.close()
        losses.append(losses_log / (step + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--dataset_path',
                        type=str,
                        default='./data',
                        help='Path where datasets will be saved')
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir",
                        type=str,
                        default="trained_models/ddpm-model-64.pth")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--loss_logs_dir", type=str, default="training_logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_model_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--use_clip_grad", action='store_true')
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")
    parser.add_argument('--fp16_precision',
                        action='store_true',
                        help='Whether to use 16-bit precision for GPU training')
    parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')
    
    # === New diffusion improvements ===
    # Prediction type
    parser.add_argument('--prediction_type',
                        type=str,
                        default='v_prediction',
                        choices=['epsilon', 'v_prediction'],
                        help='Prediction target: epsilon (noise) or v_prediction')
    
    # Loss type
    parser.add_argument('--loss_type',
                        type=str,
                        default='l1',
                        choices=['l1', 'l2'],
                        help='Loss function: l1 or l2 (mse)')
    
    # Noise schedule
    parser.add_argument('--beta_schedule',
                        type=str,
                        default='shifted_cosine',
                        choices=['cosine', 'shifted_cosine', 'linear'],
                        help='Beta schedule type')
    parser.add_argument('--beta_schedule_shift',
                        type=float,
                        default=1.0,
                        help='Shift for shifted_cosine schedule (>1 = more noise)')
    
    # Noise offset for better color distribution
    parser.add_argument('--noise_offset',
                        type=float,
                        default=0.1,
                        help='Noise offset for better dark/bright image generation (try 0.1)')
    
    # Min-SNR weighting
    parser.add_argument('--use_snr_weighting',
                        action='store_true',
                        help='Use Min-SNR-gamma loss weighting')
    parser.add_argument('--snr_gamma',
                        type=float,
                        default=5.0,
                        help='Gamma value for Min-SNR weighting')
    
    # Log-normal timestep sampling
    parser.add_argument('--use_lognormal_sampling',
                        action='store_true',
                        help='Use log-normal distribution for timestep sampling')
    parser.add_argument('--lognormal_mean',
                        type=float,
                        default=0.0,
                        help='Mean for log-normal timestep sampling')
    parser.add_argument('--lognormal_std',
                        type=float,
                        default=1.0,
                        help='Std for log-normal timestep sampling')

    args = parser.parse_args()

    if args.dataset_name is None and args.dataset_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
