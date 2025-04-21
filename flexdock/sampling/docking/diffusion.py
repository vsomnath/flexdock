from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta


@dataclass(frozen=True)
class SigmaConfig:
    # Ligand diffusion
    tr_sigma_min: float
    tr_sigma_max: float
    rot_sigma_min: float
    rot_sigma_max: float
    tor_sigma_min: float
    tor_sigma_max: float

    bb_tr_sigma: Optional[float] = None
    bb_rot_sigma: Optional[float] = None
    sidechain_tor_sigma: Optional[float] = None


@dataclass(frozen=True)
class TimeConfig:
    # Parameters used for sampling time ~ [0, 1] and for initializing schedules
    alpha: float = 1
    beta: float = 1
    bb_tr_bridge_alpha: Optional[float] = None
    bb_rot_bridge_alpha: Optional[float] = None
    sc_tor_bridge_alpha: Optional[float] = None


def sigmoid(t):
    return 1 / (1 + np.e ** (-t))


def sigmoid_schedule(t, k=10, m=0.5):
    s = lambda t: sigmoid(k * (t - m))
    return (s(t) - s(0)) / (s(1) - s(0))


def bridge_transform_t(t, alpha):
    # return np.where(t > threshold, 0, np.exp(alpha * t)-np.exp(alpha)) / (1 - np.exp(alpha))
    return (np.exp(alpha * t) - np.exp(alpha)) / (1 - np.exp(alpha))


def bridge_transform_t_torch(t, alpha):
    return (torch.exp(alpha * t) - np.exp(alpha)) / (1 - np.exp(alpha))


def t_to_sigma_individual(
    t,
    schedule_type,
    sigma_min=1.0,
    sigma_max=1.0,
    schedule_k=10,
    schedule_m=0.4,
    constant_val=0.01,
):
    if schedule_type == "exponential":
        return sigma_min ** (1 - t) * sigma_max**t
    elif schedule_type == "sigmoid":
        return (
            sigmoid_schedule(t, k=schedule_k, m=schedule_m) * (sigma_max - sigma_min)
            + sigma_min
        )
    elif schedule_type == "constant":
        return constant_val


def t_to_sigma(t_dict, args: SigmaConfig):
    t_tr, t_rot, t_tor = t_dict["tr"], t_dict["rot"], t_dict["tor"]

    tr_sigma = t_to_sigma_individual(
        t_tr, "exponential", args.tr_sigma_min, args.tr_sigma_max
    )
    rot_sigma = t_to_sigma_individual(
        t_rot, "exponential", args.rot_sigma_min, args.rot_sigma_max
    )
    tor_sigma = t_to_sigma_individual(
        t_tor, "exponential", args.tor_sigma_min, args.tor_sigma_max
    )

    if args.sidechain_tor_sigma is not None:
        t_sc_tor = t_dict["sc_tor"]
        sc_tor_sigma = t_to_sigma_individual(
            t_sc_tor,
            "constant",
            constant_val=args.sidechain_tor_sigma,
        )
    else:
        sc_tor_sigma = None

    if args.bb_tr_sigma is not None:
        t_bb_tr, t_bb_rot = t_dict["bb_tr"], t_dict["bb_rot"]
        bb_tr_sigma = t_to_sigma_individual(
            t_bb_tr,
            "constant",
            constant_val=args.bb_tr_sigma,
        )
        bb_rot_sigma = t_to_sigma_individual(
            t_bb_rot, "constant", constant_val=args.bb_rot_sigma
        )
    else:
        bb_tr_sigma = bb_rot_sigma = None

    sigma_dict = {
        "tr_sigma": tr_sigma,
        "rot_sigma": rot_sigma,
        "tor_sigma": tor_sigma,
        "sc_tor_sigma": sc_tor_sigma,
        "bb_tr_sigma": bb_tr_sigma,
        "bb_rot_sigma": bb_rot_sigma,
    }
    return sigma_dict


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == "sinusoidal":
        emb_func = lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim)
    elif embedding_type == "fourier":
        emb_func = GaussianFourierProjection(
            embedding_size=embedding_dim, scale=embedding_scale
        )
    else:
        raise NotImplementedError
    return emb_func


def get_t_schedule(
    sigma_schedule, inference_steps, inf_sched_alpha=1, inf_sched_beta=1, t_max=1
):
    if sigma_schedule == "expbeta":
        lin_max = beta.cdf(t_max, a=inf_sched_alpha, b=inf_sched_beta)
        c = np.linspace(lin_max, 0, inference_steps + 1)[:-1]
        return beta.ppf(c, a=inf_sched_alpha, b=inf_sched_beta)
    raise Exception()


def get_inverse_schedule(t, sched_alpha=1, sched_beta=1):
    return beta.ppf(t, a=sched_alpha, b=sched_beta)


def set_time(
    complex_graphs,
    t,
    t_tr,
    t_rot,
    t_tor,
    t_sidechain_tor,
    t_bb_tr,
    t_bb_rot,
    batchsize,
    all_atoms,
    device,
    include_miscellaneous_atoms=False,
):
    complex_graphs["ligand"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["ligand"].num_nodes).to(device),
    }

    complex_graphs["receptor"].node_t = {
        "tr": t_tr * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "rot": t_rot * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        "tor": t_tor * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
    }

    complex_graphs.complex_t = {
        "tr": t_tr * torch.ones(batchsize).to(device),
        "rot": t_rot * torch.ones(batchsize).to(device),
        "tor": t_tor * torch.ones(batchsize).to(device),
        "t": t * torch.ones(batchsize).to(device),
    }

    if all_atoms:
        complex_graphs["atom"].node_t = {
            "tr": t_tr * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "rot": t_rot * torch.ones(complex_graphs["atom"].num_nodes).to(device),
            "tor": t_tor * torch.ones(complex_graphs["atom"].num_nodes).to(device),
        }

    if t_sidechain_tor is not None:
        complex_graphs["ligand"].node_t["sc_tor"] = t_sidechain_tor * torch.ones(
            complex_graphs["ligand"].num_nodes
        ).to(device)
        complex_graphs["receptor"].node_t["sc_tor"] = t_sidechain_tor * torch.ones(
            complex_graphs["receptor"].num_nodes
        ).to(device)

        complex_graphs.complex_t["sc_tor"] = t_sidechain_tor * torch.ones(batchsize).to(
            device
        )

        if all_atoms:
            complex_graphs["atom"].node_t["sc_tor"] = t_sidechain_tor * torch.ones(
                complex_graphs["atom"].num_nodes
            ).to(device)

    if t_bb_tr is not None:
        complex_graphs["receptor"].node_t["bb_tr"] = (
            t_bb_tr * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
        )
        complex_graphs["receptor"].node_t["bb_rot"] = t_bb_rot * torch.ones(
            complex_graphs["receptor"].num_nodes
        ).to(device)

        complex_graphs.complex_t["bb_tr"] = t_bb_tr * torch.ones(batchsize).to(device)
        complex_graphs.complex_t["bb_rot"] = t_bb_rot * torch.ones(batchsize).to(device)

        if all_atoms:
            complex_graphs["atom"].node_t["bb_tr"] = (
                t_bb_tr * torch.ones(complex_graphs["receptor"].num_nodes).to(device),
            )
            complex_graphs["atom"].node_t["bb_rot"] = t_bb_rot * torch.ones(
                complex_graphs["receptor"].num_nodes
            ).to(device)

    # TODO: Fix this for including t_bb_tr and t_bb_rot
    if include_miscellaneous_atoms and not all_atoms:
        complex_graphs["misc_atom"].node_t = {
            "tr": t_tr * torch.ones(complex_graphs["misc_atom"].num_nodes).to(device),
            "rot": t_rot * torch.ones(complex_graphs["misc_atom"].num_nodes).to(device),
            "tor": t_tor * torch.ones(complex_graphs["misc_atom"].num_nodes).to(device),
            "sc_tor": t_sidechain_tor
            * torch.ones(complex_graphs["misc_atom"].num_nodes).to(device),
        }


def set_time_t_dict(
    complex_graphs,
    t_dict,
    batchsize,
    all_atoms,
    device,
    include_miscellaneous_atoms=False,
):
    t_tr, t_rot, t_tor, t_sidechain_tor, t = (
        t_dict["tr"],
        t_dict["rot"],
        t_dict["tor"],
        t_dict["sc_tor"],
        t_dict["t"],
    )
    t_bb_tr, t_bb_rot = t_dict["bb_tr"], t_dict["bb_rot"]
    set_time(
        complex_graphs=complex_graphs,
        t=t,
        t_tr=t_tr,
        t_rot=t_rot,
        t_tor=t_tor,
        t_sidechain_tor=t_sidechain_tor,
        t_bb_tr=t_bb_tr,
        t_bb_rot=t_bb_rot,
        batchsize=batchsize,
        all_atoms=all_atoms,
        device=device,
        include_miscellaneous_atoms=include_miscellaneous_atoms,
    )
