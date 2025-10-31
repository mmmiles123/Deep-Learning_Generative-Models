import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.utils import make_grid
from torchvision import transforms

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, device="cuda"):
        super().__init__()
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            self.sqrt_alphas_cumprod[t, None, None, None] * x0 +
            self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

    def forward(self, x0, cond):
        b, c, h, w = x0.shape
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_noisy = self.add_noise(x0, t, noise)
        noise_pred = self.model(x_noisy, t, cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, cond, image_size=64, channels=3, batch_size=1, 
               guidance=None, guidance_scale=0.0, save_intermediates=False):
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        intermediates = [x.clone()] if save_intermediates else []

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_batch, cond)

            if guidance is not None and guidance_scale > 0:
                x_normed = (x + 1) / 2
                normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                x_in = torch.stack([normalize(xi) for xi in x_normed]).detach().requires_grad_()
                with torch.enable_grad():
                    logits = guidance(x_in)
                    loss = F.binary_cross_entropy_with_logits(logits, cond)
                    grad = torch.autograd.grad(loss, x_in)[0]
                x = x - guidance_scale * grad * 2

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise)
                + torch.sqrt(beta) * noise
            ).clamp(-1, 1)

            if save_intermediates:
                intermediates.append(x.clone())

        return x, intermediates if save_intermediates else x

    def make_grid(self, images, nrow=8):
        images = (images + 1) / 2
        return make_grid(images, nrow=nrow, normalize=False)

    def make_denoise_grid(self, intermediates, nrow=8):
        images = [img[0:1] for img in intermediates]
        images = torch.cat(images, dim=0)
        return self.make_grid(images, nrow=nrow)
