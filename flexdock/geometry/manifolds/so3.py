"""
Operations and definitions of diffusion processes on SO(3) manifold
"""

import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Union

from lightning.pytorch.utilities import rank_zero_info

from geomstats.geometry.special_orthogonal import SpecialOrthogonal


# By default the manifold is on numpy
MANIFOLD = SpecialOrthogonal(n=3, point_type="vector")
MIN_EPS, MAX_EPS, N_EPS = 0.02, 4, 1000
X_N = 2000

"""
    Preprocessing for the SO(3) sampling and score computations, truncated
    infinite series are computed and then cached to memory, therefore the
    precomputation is only run the first time the repository is run on a machine
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:]


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2 / 2)
            * np.sin(omega * (l + 1 / 2))
            / np.sin(omega / 2)
        )
    return p


def _density(
    expansion, omega, marginal=True
):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return (
            expansion / 8 / np.pi**2
        )  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2 / 2)
            * (lo * dhi - hi * dlo)
            / lo**2
        )
    return dSigma / exp


if os.path.exists(".so3_omegas_array3.npy"):
    rank_zero_info("Precomputed score cache found for so3. Loading them...")
    _omegas_array = np.load(".so3_omegas_array3.npy")
    _cdf_vals = np.load(".so3_cdf_vals3.npy")
    _score_norms = np.load(".so3_score_norms3.npy")
    _exp_score_norms = np.load(".so3_exp_score_norms3.npy")
else:
    rank_zero_info("Precomputed score cache not found for so3. Precomputing...")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array])
    _pdf_vals = np.asarray(
        [_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]
    )
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray(
        [
            _score(_exp_vals[i], _omegas_array, _eps_array[i])
            for i in range(len(_eps_array))
        ]
    )

    _exp_score_norms = np.sqrt(
        np.sum(_score_norms**2 * _pdf_vals, axis=1)
        / np.sum(_pdf_vals, axis=1)
        / np.pi
    )

    np.save(".so3_omegas_array3.npy", _omegas_array)
    np.save(".so3_cdf_vals3.npy", _cdf_vals)
    np.save(".so3_score_norms3.npy", _score_norms)
    np.save(".so3_exp_score_norms3.npy", _exp_score_norms)


def sample(eps, shape=None):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    if shape is not None:
        x = np.random.rand(*shape)
    else:
        x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec_individual(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def sample_vec(eps, shape=None):
    if shape is None:
        return sample_vec_individual(eps=eps)
    x = np.random.randn(*shape)  # [..., 3]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x * sample(eps=eps, shape=shape[:-1])[..., None]


def score_vec(eps, vec):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    eps = eps.numpy()
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()


def log_map_at_point(
    point: Union[torch.Tensor, np.ndarray],
    base_point: Union[torch.Tensor, np.ndarray],
):
    if torch.is_tensor(point):
        point = point.numpy()

    if base_point is not None and torch.is_tensor(base_point):
        base_point = base_point.numpy()

    return MANIFOLD.log(point=point, base_point=base_point)


def exp_map_at_point(tangent_vec: np.ndarray, base_point: np.ndarray):
    exp_point = MANIFOLD.exp(tangent_vec=tangent_vec, base_point=base_point)
    return torch.tensor(exp_point)


def sample_from_igso3(
    mu: Union[torch.Tensor, np.ndarray], sigma: Union[torch.Tensor, np.ndarray]
):
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    if not sigma:
        if not torch.is_tensor(mu):
            return torch.from_numpy(mu)

    zero_mean_sample = sample_vec(eps=sigma, shape=mu.shape)
    composed_sample = MANIFOLD.compose(
        point_a=mu, point_b=MANIFOLD.exp_from_identity(tangent_vec=zero_mean_sample)
    )

    if not torch.is_tensor(composed_sample):
        return torch.from_numpy(composed_sample)


def compose(
    point_a: Union[torch.Tensor, np.ndarray],
    point_b: Union[torch.Tensor, np.ndarray],
):
    if torch.is_tensor(point_a):
        point_a = point_a.numpy()

    if torch.is_tensor(point_b):
        point_b = point_b.numpy()

    return MANIFOLD.compose(point_a, point_b)


"""
def compute_bridge_mu_t(x0, x1, t):
    if x0 is not None and torch.is_tensor(x0):
        x0 = x0.numpy()

    if torch.is_tensor(x1):
        x1 = x1.numpy()

    log_x1_at_x0 = MANIFOLD.log(
        point=x1, base_point=x0
    )

    mu_t = MANIFOLD.exp(
        tangent_vec=t * log_x1_at_x0,
        base_point=x0
    )

    return torch.tensor(mu_t)


def compute_bridge_std_t(t, g, clamp_value: float = None):
    sigma_t = g * np.sqrt(t * (1-t) + 1e-4)

    if clamp_value is not None:
        return torch.clamp(sigma_t, clamp_value)
    return sigma_t


def sample_from_bridge(
    x0, x1, t, g: float = 0.01,
    ode: bool = False,
    clamp_value: float = None
):
    mu_t = compute_bridge_mu_t(x0=x0, x1=x1, t=t)

    if ode:
        return mu_t
    sigma_t = compute_bridge_std_t(t=t, g=g, clamp_value=clamp_value)
    return sample_from_igso3(mu=mu_t, sigma=sigma_t)
"""
