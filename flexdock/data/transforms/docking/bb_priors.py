import numpy as np
import torch


class HarmonicPrior:
    def __init__(
        self,
        bb_random_prior_ot: int = 1,
        bb_random_prior_std: float = 0.1,
        bb_random_prior_ot_inf: int = 1,
    ):
        self.bb_random_prior_ot = bb_random_prior_ot
        self.bb_random_prior_std = bb_random_prior_std
        self.bb_random_prior_ot_inf = bb_random_prior_ot_inf

    def sample_harmonic_noise(
        self,
        n_residues: int,
        n_samples: int,
        alpha: float = 3 / (3.8**2),
        std: float = 0.1,
    ):
        J = torch.zeros(n_residues, n_residues)

        for i, j in zip(np.arange(n_residues - 1), np.arange(1, n_residues)):
            J[i, i] += alpha
            J[j, j] += alpha
            J[i, j] = J[j, i] = -alpha
        D, P = torch.linalg.eigh(J)

        D_inv = 1 / D
        D_inv[0] = 0

        if n_samples > 1:
            harmonic_noise = (
                P
                @ (torch.sqrt(D_inv)[:, None] * torch.randn(n_samples, n_residues, 3))
                * std
            )
        else:
            harmonic_noise = (
                P @ (torch.sqrt(D_inv)[:, None] * torch.randn(n_residues, 3)) * std
            )
        return harmonic_noise

    def __call__(self, calpha_apo, calpha_holo):
        if self.bb_random_prior_ot > 1:
            calpha_delta_random = self.sample_harmonic_noise(
                n_residues=calpha_apo.shape[0],
                n_samples=self.bb_random_prior_ot,
                std=self.bb_random_prior_std,
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=1, keepdim=True)
            calpha_random = calpha_apo.unsqueeze(0) + calpha_delta_random
            rmsds = (
                ((calpha_random - calpha_holo.unsqueeze(0)) ** 2)
                .sum(dim=-1)
                .mean(dim=-1)
            )
            calpha_apo = calpha_random[rmsds.argmin()]

        else:
            calpha_delta_random = self.sample_harmonic_noise(
                n_residues=calpha_apo.shape[0],
                n_samples=1,
                std=self.bb_random_prior_std,
            )

            calpha_delta_random -= calpha_delta_random.mean(dim=0, keepdim=True)
            calpha_apo = calpha_apo + calpha_delta_random

        return calpha_apo

    def sample_for_inference(self, complex_graph):
        if self.bb_random_prior_ot_inf > 1:
            calpha_holo = complex_graph["atom"].orig_holo_pos[
                complex_graph["atom"].calpha
            ]
            calpha_delta_random = self.sample_harmonic_noise(
                n_residues=complex_graph["receptor"].pos.shape[0],
                n_samples=self.bb_random_prior_ot_inf,
                std=self.bb_random_prior_std,
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=1, keepdim=True)
            calpha_random = (
                complex_graph["receptor"].pos.unsqueeze(0) + calpha_delta_random
            )
            rmsds = (
                ((calpha_random - calpha_holo.unsqueeze(0)) ** 2)
                .sum(dim=-1)
                .mean(dim=-1)
            )
            calpha_delta_random = calpha_delta_random[rmsds.argmin()]

        else:
            calpha_delta_random = self.sample_harmonic_noise(
                n_residues=complex_graph["receptor"].pos.shape[0],
                n_samples=1,
                std=self.bb_random_prior_std,
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=0, keepdim=True)

        return calpha_delta_random


class GaussianPrior:
    def __init__(
        self,
        bb_random_prior_ot: int = 1,
        bb_random_prior_std: float = 0.1,
        bb_random_prior_ot_inf: int = 1,
    ):
        self.bb_random_prior_ot = bb_random_prior_ot
        self.bb_random_prior_std = bb_random_prior_std
        self.bb_random_prior_ot_inf = bb_random_prior_ot_inf

    def __call__(self, calpha_apo, calpha_holo):
        if self.bb_random_prior_ot > 1:
            calpha_delta_random = torch.randn(
                (self.bb_random_prior_ot, calpha_apo.shape[0], calpha_apo.shape[1])
            )
            calpha_delta_random *= self.bb_random_prior_std
            calpha_delta_random -= calpha_delta_random.mean(dim=1, keepdim=True)
            calpha_random = calpha_apo.unsqueeze(0) + calpha_delta_random
            rmsds = (
                ((calpha_random - calpha_holo.unsqueeze(0)) ** 2)
                .sum(dim=-1)
                .mean(dim=-1)
            )
            calpha_apo = calpha_random[rmsds.argmin()]

        else:
            calpha_delta_random = (
                torch.randn_like(calpha_apo) * self.bb_random_prior_std
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=0, keepdim=True)
            calpha_apo = calpha_apo + calpha_delta_random

        return calpha_apo

    def sample_for_inference(self, complex_graph):
        if self.bb_random_prior_ot_inf > 1:
            calpha_holo = complex_graph["atom"].orig_holo_pos[
                complex_graph["atom"].calpha
            ]
            calpha_delta_random = (
                torch.randn(
                    (
                        self.bb_random_prior_ot_inf,
                        complex_graph["receptor"].pos.shape[0],
                        complex_graph["receptor"].pos.shape[1],
                    )
                )
                * self.bb_random_prior_std
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=1, keepdim=True)
            calpha_random = (
                complex_graph["receptor"].pos.unsqueeze(0) + calpha_delta_random
            )
            rmsds = (
                ((calpha_random - calpha_holo.unsqueeze(0)) ** 2)
                .sum(dim=-1)
                .mean(dim=-1)
            )
            calpha_delta_random = calpha_delta_random[rmsds.argmin()]

        else:
            calpha_delta_random = (
                torch.randn_like(complex_graph["receptor"].pos)
                * self.bb_random_prior_std
            )
            calpha_delta_random -= calpha_delta_random.mean(dim=0, keepdim=True)

        return calpha_delta_random


def construct_bb_prior(args):
    if not args.bb_random_prior:
        return None

    try:
        print(f"Constructing prior with noise {args.bb_random_prior_noise}")
    except Exception:
        print("Constructing prior with Gaussian noise")

    if not hasattr(args, "bb_random_prior_noise"):
        args.bb_random_prior_noise = "gaussian"

    if args.bb_random_prior_noise == "gaussian":
        prior = GaussianPrior(
            bb_random_prior_ot=args.bb_random_prior_ot,
            bb_random_prior_std=args.bb_random_prior_std,
            bb_random_prior_ot_inf=args.bb_random_prior_ot_inf,
        )

    elif args.bb_random_prior_noise == "harmonic":
        prior = HarmonicPrior(
            bb_random_prior_ot=args.bb_random_prior_ot,
            bb_random_prior_std=args.bb_random_prior_std,
            bb_random_prior_ot_inf=args.bb_random_prior_ot_inf,
        )

    else:
        print(f" Prior with noise {args.bb_random_prior_noise} not supported yet.")
        prior = None

    return prior
