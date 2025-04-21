import os

import numpy as np
import torch

from lightning.pytorch.utilities import rank_zero_info


# ==============================================================================
# Diffusion ops for the torus
# ==============================================================================


def p(x, sigma, N=10):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (
            (x + 2 * np.pi * i)
            / sigma**2
            * np.exp(-((x + 2 * np.pi * i) ** 2) / 2 / sigma**2)
        )
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

if os.path.exists(".torus_p.npy"):
    rank_zero_info("Precomputed score cache found for torus. Loading them...")
    p_ = np.load(".torus_p.npy")
    score_ = np.load(".torus_score.npy")
else:
    rank_zero_info("Precomputed score cache not found for torus. Precomputing...")
    p_ = p(x, sigma[:, None], N=100)
    np.save(".torus_p.npy", p_)

    score_ = grad(x, sigma[:, None], N=100) / p_
    np.save(".torus_score.npy", score_)


def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten(),
).reshape(10000, -1)
score_norm_ = (score_norm_**2).mean(0)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (
        (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    )
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]


def log_map_at_point(point: torch.Tensor, base_point: torch.Tensor):
    if base_point is None:
        base_point = torch.zeros_like(point)

    sin_term = torch.sin(point - base_point)
    cos_term = torch.cos(point - base_point)
    return torch.atan2(sin_term, cos_term)


def exp_map_at_point(tangent_vec: torch.Tensor, base_point: torch.Tensor):
    if base_point is None:
        base_point = torch.zeros_like(tangent_vec)

    out = (base_point + tangent_vec + torch.pi) % (2 * torch.pi) - torch.pi
    return out


def sample_from_wrapped_normal(mu: torch.Tensor, sigma: torch.Tensor):
    x = sigma * torch.randn_like(mu)
    x = mu + (x + torch.pi) % (2 * torch.pi) - torch.pi
    return x


"""
def compute_bridge_mu_t(x0, x1, t):
    log_x1_at_x0 = log_map_at_point(point=x1, base_point=x0)
    mu_t = exp_map_at_point(
        tangent_vec=t * log_x1_at_x0,
        base_point=x0
    )
    return mu_t

def compute_bridge_std_t(t, g, clamp_value: float = None):
    sigma_t = g * np.sqrt(t * (1-t) + 1e-4)
    if clamp_value is not None:
        return torch.clamp(sigma_t, clamp_value)
    return sigma_t


def sample_from_bridge(x0, x1, t, g, clamp_value: float = None, ode: bool = False):
    mu_t = compute_bridge_mu_t(x0=x0, x1=x1, t=t)
    if ode:
        return mu_t
    sigma_t = compute_bridge_std_t(t=t, g=g, clamp_value=clamp_value)
    return sample_from_wrapped_normal(mu=mu_t, sigma=sigma_t)
"""
