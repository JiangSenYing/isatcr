"""
Conditional diffusion action prior for discrete satellite routing actions.

The model treats the discrete action as a noisy continuous one-hot vector. It is
conditioned on the current Q-value vector, so it can be attached to any existing
flat/GNN controller without changing the controller internals.
"""

import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: th.Tensor) -> th.Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = th.exp(th.arange(half_dim, device=timesteps.device) * -scale)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = th.cat([th.sin(args), th.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiffusionActionPrior(nn.Module):
    """
    Denoising diffusion model over a continuous one-hot action vector.

    Args:
        n_actions: number of discrete actions.
        condition_dim: dimension of the conditioning vector.
        hidden_size: MLP hidden size.
        diffusion_steps: number of denoising steps.
        time_embed_dim: dimension of sinusoidal timestep embedding.
        beta_start/beta_end: linear noise schedule.
    """

    def __init__(
        self,
        n_actions: int,
        condition_dim: Optional[int] = None,
        hidden_size: int = 128,
        diffusion_steps: int = 20,
        time_embed_dim: int = 32,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.condition_dim = condition_dim or n_actions
        self.diffusion_steps = diffusion_steps

        betas = th.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = th.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", th.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", th.sqrt(1.0 - alpha_bars))

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_actions + self.condition_dim + time_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def predict_noise(self, noisy_action: th.Tensor, timesteps: th.Tensor, condition: th.Tensor) -> th.Tensor:
        t_emb = self.time_embedding(timesteps)
        x = th.cat([noisy_action, condition, t_emb], dim=-1)
        return self.net(x)

    def training_loss(self, condition: th.Tensor, actions: th.Tensor) -> th.Tensor:
        batch_size = actions.size(0)
        clean_action = F.one_hot(actions, num_classes=self.n_actions).float()
        timesteps = th.randint(0, self.diffusion_steps, (batch_size,), device=actions.device)
        noise = th.randn_like(clean_action)

        sqrt_ab = self.sqrt_alpha_bars[timesteps].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[timesteps].unsqueeze(-1)
        noisy_action = sqrt_ab * clean_action + sqrt_omab * noise

        pred_noise = self.predict_noise(noisy_action, timesteps, condition)
        return F.mse_loss(pred_noise, noise)

    def denoise_logits(self, condition: th.Tensor) -> th.Tensor:
        """
        Differentiable deterministic denoising pass used for Q-guided training.
        """
        x = th.zeros(condition.size(0), self.n_actions, device=condition.device)
        for step in reversed(range(self.diffusion_steps)):
            timesteps = th.full((condition.size(0),), step, device=condition.device, dtype=th.long)
            pred_noise = self.predict_noise(x, timesteps, condition)

            alpha = self.alphas[step]
            beta = self.betas[step]
            alpha_bar = self.alpha_bars[step]
            x = (x - beta / th.sqrt(1.0 - alpha_bar) * pred_noise) / th.sqrt(alpha)
        return x

    @th.no_grad()
    def sample_logits(self, condition: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Sample a denoised action vector. The result is used as action-prior logits.
        """
        if deterministic:
            return self.denoise_logits(condition)
        x = th.randn(condition.size(0), self.n_actions, device=condition.device)
        for step in reversed(range(self.diffusion_steps)):
            timesteps = th.full((condition.size(0),), step, device=condition.device, dtype=th.long)
            pred_noise = self.predict_noise(x, timesteps, condition)

            alpha = self.alphas[step]
            beta = self.betas[step]
            alpha_bar = self.alpha_bars[step]
            x = (x - beta / th.sqrt(1.0 - alpha_bar) * pred_noise) / th.sqrt(alpha)

            if step > 0 and not deterministic:
                x = x + th.sqrt(beta) * th.randn_like(x)
        return x


class DiffusionGuidedQNetwork(nn.Module):
    """
    Inference wrapper that adds a learned diffusion prior to Q-values.

    It mirrors the wrapped controller's forward return type:
    - tensor in, tensor out for flat controllers
    - tuple(q_values, hidden) for graph controllers
    """

    def __init__(
        self,
        q_network: nn.Module,
        diffusion_prior: DiffusionActionPrior,
        guidance_weight: float,
        deterministic: bool = True,
        normalize_prior: bool = True,
        prior_clip: float = 2.0,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.q_network = q_network
        self.diffusion_prior = diffusion_prior
        self.guidance_weight = guidance_weight
        self.deterministic = deterministic
        self.normalize_prior = normalize_prior
        self.prior_clip = prior_clip
        self.guidance_scale = guidance_scale

    def set_guidance_scale(self, guidance_scale: float):
        self.guidance_scale = max(0.0, min(1.0, guidance_scale))

    def forward(self, *args, **kwargs):
        output = self.q_network(*args, **kwargs)
        if isinstance(output, tuple):
            q_values, *rest = output
            guided = self._guide(q_values)
            return (guided, *rest)
        return self._guide(output)

    def _guide(self, q_values: th.Tensor) -> th.Tensor:
        effective_weight = self.guidance_weight * self.guidance_scale
        if effective_weight == 0:
            return q_values
        prior_logits = self.diffusion_prior.sample_logits(q_values.detach(), deterministic=self.deterministic)
        if self.normalize_prior:
            prior_logits = (prior_logits - prior_logits.mean(dim=-1, keepdim=True)) / (
                prior_logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
            )
        if self.prior_clip > 0:
            prior_logits = prior_logits.clamp(-self.prior_clip, self.prior_clip)
        return q_values + effective_weight * prior_logits
