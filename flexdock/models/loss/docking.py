import numpy as np
import torch
import torch.nn as nn

from flexdock.geometry.manifolds import so3, torus


class FlexDockLoss(nn.Module):
    def __init__(self, args, t_to_sigma):
        super().__init__()
        self.args = args
        self.t_to_sigma = t_to_sigma
        self._prepare_loss_weights()

    def _prepare_loss_weights(self):
        self.loss_weights = {
            "tr_loss": self.args.tr_weight,
            "rot_loss": self.args.rot_weight,
            "tor_loss": self.args.tor_weight,
            "bb_tr_loss": self.args.bb_tr_weight,
            "bb_rot_loss": self.args.bb_rot_weight,
            "sc_tor_loss": self.args.sc_tor_weight,
        }

    def forward(self, outputs, batch, apply_mean: bool = False):
        loss_dict = {}

        noise_types = ["tr", "rot", "tor"]

        if self.args.flexible_sidechains:
            noise_types.append("sc_tor")

        # This is only for compatibility. For backbone bridge, t is not used for sigma
        if self.args.flexible_backbone:
            noise_types.extend(["bb_tr", "bb_rot"])

        noise_types.append("t")

        if isinstance(batch, list):
            t_dict = {
                noise_type: torch.cat([d.complex_t[noise_type] for d in batch])
                for noise_type in noise_types
            }
        else:
            t_dict = {
                noise_type: batch.complex_t[noise_type] for noise_type in noise_types
            }

        sigma_dict = self.t_to_sigma(t_dict)

        ligand_losses = self.compute_ligand_loss(
            outputs, batch, t_dict, sigma_dict, apply_mean=apply_mean
        )
        protein_losses = self.compute_protein_loss(
            outputs, batch, t_dict, sigma_dict, apply_mean=apply_mean
        )

        for loss_name, value in ligand_losses.items():
            loss_dict[loss_name] = value.detach().clone()

        for loss_name, value in protein_losses.items():
            loss_dict[loss_name] = value.detach().clone()

        loss = 0.0
        for loss_name, value in ligand_losses.items():
            if "base_loss" in loss_name:
                continue
            loss = loss + self.loss_weights[loss_name] * value

        for loss_name, value in protein_losses.items():
            if "base_loss" in loss_name:
                continue
            loss = loss + self.loss_weights[loss_name] * value

        loss_dict["loss"] = loss.detach().clone()
        return loss, loss_dict

    def compute_ligand_loss(
        self, outputs, batch, t_dict, sigma_dict, apply_mean: bool = False
    ):
        mean_dims = (0, 1) if apply_mean else 1

        pred_ = outputs["tr_pred"]

        tr_sigma = sigma_dict["tr_sigma"].to(pred_.device)
        rot_sigma = sigma_dict["rot_sigma"].to(pred_.device)

        ligand_losses = {}

        tr_pred, rot_pred = outputs["tr_pred"], outputs["rot_pred"]
        if not self.args.no_torsion:
            tor_pred = outputs["tor_pred"]

        if self.args.lig_transform_type == "flow":
            raise NotImplementedError()
            tr_flow = (
                torch.cat([d.tr_flow for d in batch], dim=0)
                if isinstance(batch, list)
                else batch.tr_flow
            )
            tr_loss = ((tr_pred - tr_flow) ** 2).mean(dim=mean_dims)

            rot_flow = (
                torch.cat([d.rot_flow for d in batch], dim=0)
                if isinstance(batch, list)
                else batch.rot_flow
            )
            rot_loss = ((rot_pred - rot_flow) ** 2).mean(dim=mean_dims)

            if not self.args.no_torsion:
                tor_flow = (
                    torch.cat([d.tor_flow for d in batch], dim=0)
                    if isinstance(batch, list)
                    else batch.tor_flow
                )
                tor_loss = (tor_pred - tor_flow) ** 2

        elif self.args.lig_transform_type == "diffusion":
            # translation component
            tr_score = batch.tr_score
            tr_sigma = tr_sigma.unsqueeze(-1)

            tr_loss = (tr_pred - tr_score) ** 2 * tr_sigma**2
            tr_base_loss = tr_score**2 * tr_sigma**2

            if hasattr(batch, "loss_weight"):
                tr_loss = tr_loss * batch.loss_weight.unsqueeze(-1)
                tr_base_loss = tr_base_loss * batch.loss_weight.unsqueeze(-1)

                # The multiplication with 3 is because of 3 dimensions
                tr_loss = tr_loss.sum() / (
                    tr_score.size(1) * batch.loss_weight.sum() + 0.0001
                )
                tr_base_loss = tr_base_loss.sum() / (
                    tr_score.size(1) * batch.loss_weight.sum() + 0.0001
                )

            else:
                tr_loss = tr_loss.mean(dim=mean_dims)
                tr_base_loss = tr_base_loss.mean(dim=mean_dims)

            # Rotation component for loss
            rot_score = batch.rot_score
            rot_score_norm = (
                so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).to(pred_.device)
            )

            rot_loss = ((rot_pred - rot_score) / rot_score_norm) ** 2
            rot_base_loss = (rot_score / rot_score_norm) ** 2

            if hasattr(batch, "loss_weight"):
                rot_loss = rot_loss * batch.loss_weight.unsqueeze(-1)
                rot_base_loss = rot_base_loss * batch.loss_weight.unsqueeze(-1)

                # size is used here to multiply zero across all dimensions
                rot_loss = rot_loss.sum() / (
                    rot_score.size(1) * batch.loss_weight.sum() + 0.0001
                )
                rot_base_loss = rot_base_loss.sum() / (
                    rot_score.size(1) * batch.loss_weight.sum() + 0.0001
                )

            else:
                rot_loss = rot_loss.mean(dim=mean_dims)
                rot_base_loss = rot_base_loss.mean(dim=mean_dims)

            ligand_losses["tr_base_loss"] = tr_base_loss
            ligand_losses["rot_base_loss"] = rot_base_loss

            if not self.args.no_torsion:
                edge_tor_sigma = torch.from_numpy(np.concatenate(batch.tor_sigma_edge))

                tor_score = batch.tor_score
                tor_score_norm2 = (
                    torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy()))
                    .float()
                    .to(pred_.device)
                )
                tor_loss = (tor_pred - tor_score) ** 2 / tor_score_norm2
                tor_base_loss = ((tor_score**2 / tor_score_norm2)).detach()

                index = batch["ligand"].batch[
                    batch["ligand", "lig_bond", "ligand"].edge_index[0][
                        batch["ligand"].edge_mask
                    ]
                ]

                if apply_mean:
                    if hasattr(batch, "loss_weight"):
                        loss_weight_tor = batch.loss_weight[index]
                        tor_loss = tor_loss * loss_weight_tor
                        tor_base_loss = tor_base_loss * loss_weight_tor

                        tor_loss = tor_loss.sum() / (loss_weight_tor.sum() + 0.0001)
                        tor_base_loss = tor_base_loss.sum() / (
                            loss_weight_tor.sum() + 0.0001
                        )

                    else:
                        tor_loss = tor_loss.mean() * rot_loss.new_ones(
                            1, dtype=torch.float
                        )
                        tor_base_loss = tor_base_loss.mean() * rot_loss.new_ones(
                            1, dtype=torch.float
                        )
                else:
                    num_graphs = batch.num_graphs
                    t_l, t_b_l, c = (
                        rot_loss.new_zeros(num_graphs),
                        rot_loss.new_zeros(num_graphs),
                        rot_loss.new_zeros(num_graphs),
                    )

                    if hasattr(batch, "loss_weight"):
                        loss_weight_tor = batch.loss_weight[index]
                        c.index_add_(
                            0,
                            index,
                            rot_loss.new_ones(tor_loss.shape) * loss_weight_tor,
                        )
                    else:
                        c.index_add_(0, index, rot_loss.new_ones(tor_loss.shape))

                    t_l.index_add_(0, index, tor_loss)
                    t_b_l.index_add_(0, index, tor_base_loss)
                    c = c + 0.0001

                    if hasattr(batch, "loss_weight"):
                        t_l = t_l * batch.loss_weight
                        t_b_l = t_b_l * batch.loss_weight
                    tor_loss, tor_base_loss = t_l / c, t_b_l / c
            else:
                if apply_mean:
                    tor_loss = rot_loss.new_zeros(1, dtype=torch.float)
                    tor_base_loss = rot_loss.new_zeros(1, dtype=torch.float)
                else:
                    tor_loss = rot_loss.new_zeros(len(rot_loss), dtype=torch.float)
                    tor_base_loss = rot_loss.new_zeros(len(rot_loss), dtype=torch.float)

            ligand_losses["tor_base_loss"] = tor_base_loss

        ligand_losses["tr_loss"] = tr_loss
        ligand_losses["rot_loss"] = rot_loss
        ligand_losses["tor_loss"] = tor_loss

        return ligand_losses

    def compute_protein_loss(
        self, outputs, batch, t_dict, sigma_dict, apply_mean: bool = False
    ):
        protein_losses = {}
        sc_tor_pred = outputs["sc_tor_pred"]

        if self.args.flexible_sidechains:
            sc_edge_tor_sigma = torch.from_numpy(
                np.concatenate(batch.sidechain_tor_sigma_edge)
            )
            sc_tor_score = batch.sidechain_tor_score

            sc_tor_score_norm2 = torch.tensor(
                torus.score_norm(sc_edge_tor_sigma.cpu().numpy()),
                device=sc_tor_pred.device,
            ).float()
            sc_tor_loss = (sc_tor_pred - sc_tor_score) ** 2 / sc_tor_score_norm2
            sc_tor_base_loss = ((sc_tor_score**2 / sc_tor_score_norm2)).detach()

            if getattr(self.args, "use_new_pipeline", False):
                bonds = batch["atom", "atom_bond", "atom"].edge_index[
                    :, batch["atom", "atom_bond", "atom"].edge_mask
                ]
                index = batch["atom"].orig_batch[bonds[0]]
            else:
                _, atom_bin_counts = batch["atom"].batch.unique(
                    sorted=True, return_counts=True
                )
                bond_offset = atom_bin_counts.cumsum(dim=0)
                # shift it by one so to speak, because the first batch does not have an offset
                bond_offset = (
                    torch.cat(
                        (
                            sc_tor_pred.new_zeros(1, device=bond_offset.device),
                            bond_offset,
                        )
                    )[:-1]
                ).long()

                # store the bonds of the flexible residues. i.e., we store which atoms are connected
                index = batch["atom"].batch[bond_offset[batch["flexResidues"].batch]]

            if hasattr(batch, "loss_weight"):
                loss_weight_sc_tor = batch.loss_weight[index]
                sc_tor_loss = sc_tor_loss * loss_weight_sc_tor
                sc_tor_base_loss = sc_tor_base_loss * loss_weight_sc_tor

            if apply_mean:
                if hasattr(batch, "loss_weight"):
                    sc_tor_loss = sc_tor_loss.sum() / (
                        loss_weight_sc_tor.sum() + 0.0001
                    )
                    sc_tor_base_loss = sc_tor_base_loss.sum() / (
                        loss_weight_sc_tor.sum() + 0.0001
                    )
                else:
                    sc_tor_loss = sc_tor_loss.mean() * sc_tor_pred.new_ones(
                        1, dtype=torch.float
                    )
                    sc_tor_base_loss = sc_tor_base_loss.mean() * sc_tor_pred.new_ones(
                        1, dtype=torch.float
                    )
            else:
                num_graphs = batch.num_graphs
                t_l, t_b_l, c = (
                    sc_tor_pred.new_zeros(num_graphs),
                    sc_tor_pred.new_zeros(num_graphs),
                    sc_tor_pred.new_zeros(num_graphs),
                )

                if hasattr(batch, "loss_weight"):
                    c.index_add_(
                        0,
                        index,
                        sc_tor_pred.new_ones(sc_tor_loss.shape)
                        * batch.loss_weight[index],
                    )
                else:
                    c.index_add_(0, index, sc_tor_pred.new_ones(sc_tor_loss.shape))
                c = c + 0.0001
                t_l.index_add_(0, index, sc_tor_loss)
                t_b_l.index_add_(0, index, sc_tor_base_loss)

                if hasattr(batch, "loss_weight"):
                    t_l = t_l * batch.loss_weight
                    t_b_l = t_b_l * batch.loss_weight

                sc_tor_loss, sc_tor_base_loss = t_l / c, t_b_l / c
        else:
            if apply_mean:
                sc_tor_loss = sc_tor_pred.new_zeros(1, dtype=torch.float)
                sc_tor_base_loss = sc_tor_pred.new_zeros(1, dtype=torch.float)
            else:
                sc_tor_loss, sc_tor_base_loss = sc_tor_pred.new_zeros(
                    1, dtype=torch.float
                ), sc_tor_pred.new_zeros(
                    1, dtype=torch.float
                )  # ?

        if self.args.flexible_backbone:
            bb_tr_pred, bb_rot_pred = outputs["bb_tr_pred"], outputs["bb_rot_pred"]
            bb_tr_drift = batch.bb_tr_drift
            bb_rot_drift = batch.bb_rot_drift

            bb_tr_bridge_norm, bb_rot_bridge_norm = 1, 1

            if len(bb_tr_pred.shape) == len(bb_tr_drift.shape) + 1:
                bb_tr_pred = torch.transpose(bb_tr_pred, 0, 1)
                bb_rot_pred = torch.transpose(bb_rot_pred, 0, 1)

            bb_tr_loss = ((bb_tr_pred - bb_tr_drift) ** 2).sum(
                -1
            ).float() * bb_tr_bridge_norm
            bb_tr_base_loss = (bb_tr_drift**2).sum(-1).float() * bb_tr_bridge_norm
            bb_rot_loss = ((bb_rot_pred - bb_rot_drift) ** 2).sum(
                -1
            ).float() * bb_rot_bridge_norm
            bb_rot_base_loss = (bb_rot_drift**2).sum(-1).float() * bb_rot_bridge_norm

            if len(bb_tr_loss.shape) == len(bb_tr_base_loss.shape) + 1:
                bb_tr_loss = bb_tr_loss.mean(0)
                bb_rot_loss = bb_rot_loss.mean(0)

            if hasattr(batch, "loss_weight"):
                loss_weights = batch.loss_weight[batch["receptor"].batch]
                bb_tr_loss = bb_tr_loss * loss_weights
                bb_tr_base_loss = bb_tr_base_loss * loss_weights

                bb_rot_loss = bb_rot_loss * loss_weights
                bb_rot_base_loss = bb_rot_base_loss * loss_weights

            if apply_mean:
                if hasattr(batch, "loss_weight"):
                    bb_tr_loss = bb_tr_loss.sum() / (loss_weights.sum() + 0.0001)
                    bb_tr_base_loss = bb_tr_base_loss.sum() / (
                        loss_weights.sum() + 0.0001
                    )

                    bb_rot_loss = bb_rot_loss.sum() / (loss_weights.sum() + 0.0001)
                    bb_rot_base_loss = bb_rot_base_loss.sum() / (
                        loss_weights.sum() + 0.0001
                    )
                else:
                    bb_tr_loss = bb_tr_loss.mean() * bb_tr_pred.new_ones(
                        1, dtype=torch.float
                    )
                    bb_tr_base_loss = bb_tr_base_loss.mean() * bb_tr_pred.new_ones(
                        1, dtype=torch.float
                    )
                    bb_rot_loss = bb_rot_loss.mean() * bb_tr_pred.new_ones(
                        1, dtype=torch.float
                    )
                    bb_rot_base_loss = bb_rot_base_loss.mean() * bb_tr_pred.new_ones(
                        1, dtype=torch.float
                    )
            else:
                index = batch["receptor"].batch
                num_graphs = (
                    len(batch)
                    if isinstance(batch, list) == "cuda"
                    else batch.num_graphs
                )

                bb_tr_l, bb_tr_b_l, c_tr = (
                    bb_tr_pred.new_zeros(num_graphs),
                    bb_tr_pred.new_zeros(num_graphs),
                    bb_tr_pred.new_zeros(num_graphs),
                )
                if hasattr(batch, "loss_weight"):
                    loss_weights = batch.loss_weight[index]
                    c_tr.index_add_(
                        0, index, bb_tr_pred.new_ones(index.shape) * loss_weights
                    )
                else:
                    c_tr.index_add_(0, index, bb_tr_pred.new_ones(index.shape))
                c_tr = c_tr + 0.0001
                bb_tr_l.index_add_(0, index, bb_tr_loss)
                bb_tr_b_l.index_add_(0, index, bb_tr_base_loss)

                if hasattr(batch, "loss_weight"):
                    bb_tr_l = bb_tr_l * batch.loss_weight
                    bb_tr_b_l = bb_tr_b_l * batch.loss_weight

                bb_tr_loss, bb_tr_base_loss = bb_tr_l / c_tr, bb_tr_b_l / c_tr

                bb_rot_l, bb_rot_b_l, c_rot = (
                    bb_tr_pred.new_zeros(num_graphs),
                    bb_tr_pred.new_zeros(num_graphs),
                    bb_tr_pred.new_zeros(num_graphs),
                )
                if hasattr(batch, "loss_weight"):
                    loss_weights = batch.loss_weight[index]
                    c_rot.index_add_(
                        0, index, bb_rot_pred.new_ones(index.shape) * loss_weights
                    )
                else:
                    c_rot.index_add_(0, index, bb_rot_pred.new_ones(index.shape))
                c_rot = c_rot + 0.0001
                bb_rot_l.index_add_(0, index, bb_rot_loss)
                bb_rot_b_l.index_add_(0, index, bb_rot_base_loss)

                if hasattr(batch, "loss_weight"):
                    bb_rot_l = bb_rot_l * batch.loss_weight
                    bb_rot_b_l = bb_rot_b_l * batch.loss_weight

                bb_rot_loss, bb_rot_base_loss = bb_rot_l / c_rot, bb_rot_b_l / c_rot

        else:
            if apply_mean:
                bb_tr_loss = bb_rot_loss = sc_tor_pred.new_zeros(1, dtype=torch.float)
                bb_tr_base_loss = bb_rot_base_loss = sc_tor_pred.new_zeros(
                    1, dtype=torch.float
                )
            else:
                bb_tr_loss = bb_rot_loss = sc_tor_pred.new_zeros(
                    outputs["rot_pred"].size(0), dtype=torch.float
                )
                bb_tr_base_loss = bb_rot_base_loss = sc_tor_pred.new_zeros(
                    outputs["rot_pred"].size(0), dtype=torch.float
                )

        protein_losses["bb_tr_loss"] = bb_tr_loss
        protein_losses["bb_rot_loss"] = bb_rot_loss
        protein_losses["sc_tor_loss"] = sc_tor_loss

        return protein_losses
