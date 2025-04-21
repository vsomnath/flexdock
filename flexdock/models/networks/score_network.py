from typing import Optional, Literal
import traceback
import logging

from e3nn import o3
import torch
from e3nn.o3 import Linear
from torch import nn
from torch_cluster import radius, radius_graph, knn_graph
from torch_scatter import scatter_mean
import numpy as np
from lightning.pytorch.utilities import rank_zero_info

from flexdock.data.constants import (
    lig_feature_dims,
    rec_residue_feature_dims,
    rec_atom_feature_dims,
)
from flexdock.geometry.manifolds import so3, torus
from flexdock.models.layers.tensor_product import TensorProductConvLayer, get_irrep_seq
from flexdock.models.networks.encoders import AtomEncoder, GaussianSmearing
from flexdock.models.tensor_ops import clamped_norm


class TensorProductScoreModel(torch.nn.Module):
    def __init__(
        self,
        t_to_sigma,
        timestep_emb_func,
        in_lig_edge_features: int = 5,  # 5 now because of dative bond
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        ns: int = 16,
        nv: int = 4,
        num_conv_layers: int = 2,
        lig_max_radius: float = 5.0,
        rec_max_radius: float = 30.0,
        cross_max_distance: float = 250.0,
        center_max_distance: float = 30.0,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        no_torsion: bool = False,
        scale_by_sigma: bool = True,
        use_second_order_repr: bool = False,
        batch_norm: bool = True,
        norm_type: Optional[Literal["batch_norm", "layer_norm"]] = None,
        dynamic_max_cross: bool = False,
        dropout: float = 0.0,
        smooth_edges: bool = False,
        odd_parity: bool = False,
        separate_noise_schedule: bool = False,
        lm_embedding_type: Optional[str] = None,
        confidence_mode: bool = False,
        confidence_dropout: float = 0.0,
        confidence_no_batchnorm: bool = False,
        asyncronous_noise_schedule: bool = False,
        num_confidence_outputs: int = 1,
        fixed_center_conv: bool = False,
        no_aminoacid_identities: bool = False,
        flexible_sidechains: bool = False,
        flexible_backbone: bool = False,
        differentiate_convolutions: bool = True,
        tp_weights_layers: int = 2,
        reduce_pseudoscalars: bool = False,
        c_alpha_radius: int = 20,
        c_alpha_max_neighbors: Optional[int] = None,
        atom_radius: int = 5,
        atom_max_neighbors: Optional[int] = None,
        sidechain_tor_bridge: bool = False,
        use_bb_orientation_feats: bool = False,
        only_nearby_residues_atomic: bool = False,
        new_confidence_version: bool = False,
        atom_lig_confidence: bool = False,
        activation_func: str = "ReLU",
        norm_affine: bool = True,
        clamped_norm_min: float = 1.0e-6,
        **kwargs,
    ):
        super().__init__()
        assert (not no_aminoacid_identities) or (
            lm_embedding_type is None
        ), "no language model emb without identities"
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        sigma_embed_dim *= 3 if separate_noise_schedule else 1
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.no_torsion = no_torsion
        self.smooth_edges = smooth_edges
        self.odd_parity = odd_parity
        self.num_conv_layers = num_conv_layers
        self.timestep_emb_func = timestep_emb_func
        self.separate_noise_schedule = separate_noise_schedule
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers
        self.asyncronous_noise_schedule = asyncronous_noise_schedule
        self.fixed_center_conv = fixed_center_conv
        self.no_aminoacid_identities = no_aminoacid_identities
        self.flexible_sidechains = flexible_sidechains
        self.flexible_backbone = flexible_backbone
        self.differentiate_convolutions = differentiate_convolutions
        self.c_alpha_radius = c_alpha_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors
        self.sidechain_tor_bridge = sidechain_tor_bridge
        self.use_bb_orientation_feats = use_bb_orientation_feats
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.new_confidence_version = new_confidence_version
        self.atom_lig_confidence = atom_lig_confidence
        self.clamped_norm_min = clamped_norm_min

        activation_func = (
            activation_func.lower()
        )  # to make sure that we can pass the activation_option in any format
        rank_zero_info(f"INFO: Activation function={activation_func}")
        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(
                f"Activation function {activation_func} option not supported yet!"
            )

        self.final_layer_activation = nn.Tanh()

        # embedding layers
        self.lig_node_embedding = AtomEncoder(
            emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim
        )
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.rec_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=sigma_embed_dim,
            lm_embedding_type=lm_embedding_type,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.atom_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_atom_feature_dims,
            sigma_embed_dim=sigma_embed_dim,
        )
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.lig_distance_expansion = GaussianSmearing(
            0.0, lig_max_radius, distance_embed_dim
        )
        self.rec_distance_expansion = GaussianSmearing(
            0.0, rec_max_radius, distance_embed_dim
        )
        self.cross_distance_expansion = GaussianSmearing(
            0.0, cross_max_distance, cross_distance_embed_dim
        )

        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)

        # convolutional layers
        conv_layers = []
        off_set = 0
        for i in range(num_conv_layers):
            if self.use_bb_orientation_feats and i == 0:
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)] + " + 2x1o"
                off_set = 1
            else:
                in_irreps = irrep_seq[min(i + off_set, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1 + off_set, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                norm_type=norm_type,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else 9,
                norm_affine=norm_affine,
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        # confidence and affinity prediction layers
        if self.confidence_mode:
            output_confidence_dim = num_confidence_outputs

            confidence_input = 2 * self.ns if num_conv_layers >= 3 else self.ns
            if self.atom_lig_confidence:
                atom_confidence_input = 2 * self.ns if num_conv_layers >= 3 else self.ns
            # In the case of flexible sidechains, we also add the atom node embedding as input
            # TDOO: Fix this
            if self.new_confidence_version:
                confidence_input *= 3

            self.confidence_predictor = nn.Sequential(
                nn.Linear(confidence_input, ns),
                nn.BatchNorm1d(ns, affine=norm_affine)
                if not confidence_no_batchnorm
                else nn.Identity(),  # original
                # nn.BatchNorm1d(ns, affine=False)
                # if not confidence_no_batchnorm
                # else nn.Identity(),  # NOTE: to enable FSDP with auto-wrap-policy
                self.activation,
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns, affine=norm_affine)
                if not confidence_no_batchnorm
                else nn.Identity(),  # original
                # nn.BatchNorm1d(ns, affine=False)
                # if not confidence_no_batchnorm
                # else nn.Identity(),  # NOTE: to enable FSDP with auto-wrap-policy
                self.activation,
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, output_confidence_dim),
            )

            if self.atom_lig_confidence:
                self.atom_confidence_predictor = nn.Sequential(
                    nn.Linear(atom_confidence_input, ns),
                    nn.BatchNorm1d(ns, affine=norm_affine)
                    if not confidence_no_batchnorm
                    else nn.Identity(),
                    self.activation,
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, ns),
                    nn.BatchNorm1d(ns, affine=norm_affine)
                    if not confidence_no_batchnorm
                    else nn.Identity(),
                    self.activation,
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, output_confidence_dim),
                )
        else:
            # convolution for translational and rotational scores
            self.center_distance_expansion = GaussianSmearing(
                0.0, center_max_distance, distance_embed_dim
            )
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps="2x1o + 2x1e" if not self.odd_parity else "1x1o + 1x1e",
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                norm_type=norm_type,
                faster=sh_lmax == 1 and not use_second_order_repr,
                norm_affine=norm_affine,
            )

            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                self.activation,
                nn.Linear(ns, 1),
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                self.activation,
                nn.Linear(ns, 1),
            )

            if not no_torsion:
                # convolution for torsional score
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e"
                    if not self.odd_parity
                    else f"{ns}x0o",
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    norm_affine=norm_affine,
                    norm_type=norm_type,
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not self.odd_parity else ns, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False),
                )

            if flexible_sidechains:
                # convolution for sidechain torsional score
                self.sidechain_final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    self.activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                )

                self.final_tp_sc_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.sc_tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_sc_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e"
                    if not self.odd_parity
                    else f"{ns}x0o",
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_affine=norm_affine,
                )
                self.sc_tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not self.odd_parity else ns, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False),
                )

            if flexible_backbone:
                self.bb_o3_linear = Linear(
                    irreps_in=self.conv_layers[-1].out_irreps,
                    irreps_out="2x1o + 2x1e",
                    internal_weights=True,
                    shared_weights=True,
                )
                self.bb_tr_final_layer = nn.Sequential(
                    nn.Linear(1 + sigma_embed_dim, ns),
                    nn.Dropout(dropout),
                    self.activation,
                    nn.Linear(ns, 1),
                )
                self.bb_rot_final_layer = nn.Sequential(
                    nn.Linear(1 + sigma_embed_dim, ns),
                    nn.Dropout(dropout),
                    self.activation,
                    nn.Linear(ns, 1),
                )
        rank_zero_info(f"Unused arguments: {kwargs}")

    def forward(self, data, fast_updates: bool = False):
        if self.no_aminoacid_identities:
            data["receptor"].x = data["receptor"].x * 0

        data["atom"].orig_batch = data["atom"].batch

        if self.only_nearby_residues_atomic:
            nearby_atoms = data["atom"].nearby_atoms
            data["atom"].ca_mask = data["atom"].ca_mask[nearby_atoms]
            data["atom"].n_mask = data["atom"].n_mask[nearby_atoms]
            data["atom"].c_mask = data["atom"].c_mask[nearby_atoms]

            atom_new_idx_map = (
                torch.zeros(data["atom"].x.shape[0], dtype=torch.long).to(
                    nearby_atoms.device
                )
                + 1000000000
            )
            atom_new_idx_map[data["atom"].nearby_atoms] = torch.arange(
                len(atom_new_idx_map[nearby_atoms])
            ).to(nearby_atoms.device)
            data["atom"].atom_new_idx_map = atom_new_idx_map

            data["atom"].x = data["atom"].x[nearby_atoms]
            data["atom"].pos = data["atom"].pos[nearby_atoms]
            data["atom"].batch = data["atom"].batch[nearby_atoms]
            data["atom"].node_t["tr"] = data["atom"].node_t["tr"][nearby_atoms]

            atom_to_res_mapping = data["atom", "receptor"].edge_index[1][nearby_atoms]
            atom_res_edge_index = torch.stack(
                [
                    torch.arange(len(data["atom"].x)).to(nearby_atoms.device),
                    atom_to_res_mapping,
                ]
            )
            data["atom", "receptor"].edge_index = atom_res_edge_index

        if not self.confidence_mode:
            noise_types = ["tr", "rot", "tor"]
            if self.flexible_sidechains:
                noise_types += ["sc_tor"]

            if self.flexible_backbone:
                noise_types += ["bb_tr", "bb_rot"]
            noise_types += ["t"]

            t_dict = {
                noise_type: data.complex_t[noise_type] for noise_type in noise_types
            }

            sigma_dict = self.t_to_sigma(t_dict)
            tr_sigma = sigma_dict["tr_sigma"]
            rot_sigma = sigma_dict["rot_sigma"]
            tor_sigma = sigma_dict["tor_sigma"]
            sidechain_tor_sigma = sigma_dict["sc_tor_sigma"]

        else:
            tr_sigma, rot_sigma, tor_sigma, sidechain_tor_sigma = [
                data.complex_t[noise_type]
                for noise_type in ["tr", "rot", "tor", "sc_tor"]
            ]

        # build ligand graph
        (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
            lig_edge_weight,
        ) = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        (
            rec_node_attr,
            rec_edge_index,
            rec_edge_attr,
            rec_edge_sh,
            rec_edge_weight,
        ) = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build atom graph
        (
            atom_node_attr,
            atom_edge_index,
            atom_edge_attr,
            atom_edge_sh,
            atom_edge_weight,
        ) = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        # build cross graph
        cross_cutoff = (
            (tr_sigma * 3 + 20).unsqueeze(1)
            if self.dynamic_max_cross
            else self.cross_max_distance
        )
        (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            lr_edge_weight,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            la_edge_weight,
            ar_edge_index,
            ar_edge_attr,
            ar_edge_sh,
            ar_edge_weight,
        ) = self.build_cross_conv_graph(data, cross_cutoff)
        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

        n_lig, n_rec = len(lig_node_attr), len(rec_node_attr)

        if self.use_bb_orientation_feats:
            # add orientation features to node attributes
            rec_node_attr = torch.cat(
                [rec_node_attr, data["receptor"].bb_orientation], dim=1
            )
            lig_node_attr = torch.cat(
                [
                    lig_node_attr,
                    torch.zeros((len(lig_node_attr), 6)).to(lig_node_attr.device),
                ],
                dim=1,
            )
            atom_node_attr = torch.cat(
                [
                    atom_node_attr,
                    torch.zeros((len(atom_node_attr), 6)).to(lig_node_attr.device),
                ],
                dim=1,
            )

        node_attr = torch.cat([lig_node_attr, rec_node_attr, atom_node_attr], dim=0)
        rec_edge_index, atom_edge_index, lr_edge_index, la_edge_index, ar_edge_index = (
            rec_edge_index.clone(),
            atom_edge_index.clone(),
            lr_edge_index.clone(),
            la_edge_index.clone(),
            ar_edge_index.clone(),
        )
        rec_edge_index[0], rec_edge_index[1] = (
            rec_edge_index[0] + n_lig,
            rec_edge_index[1] + n_lig,
        )
        atom_edge_index[0], atom_edge_index[1] = (
            atom_edge_index[0] + n_lig + n_rec,
            atom_edge_index[1] + n_lig + n_rec,
        )
        lr_edge_index[1] = lr_edge_index[1] + n_lig
        la_edge_index[1] = la_edge_index[1] + n_lig + n_rec
        ar_edge_index[0], ar_edge_index[1] = (
            ar_edge_index[0] + n_lig + n_rec,
            ar_edge_index[1] + n_lig,
        )

        edge_index = torch.cat(
            [
                lig_edge_index,
                lr_edge_index,
                la_edge_index,
                rec_edge_index,
                torch.flip(lr_edge_index, dims=[0]),
                torch.flip(ar_edge_index, dims=[0]),
                atom_edge_index,
                torch.flip(la_edge_index, dims=[0]),
                ar_edge_index,
            ],
            dim=1,
        )
        edge_attr = torch.cat(
            [
                lig_edge_attr,
                lr_edge_attr,
                la_edge_attr,
                rec_edge_attr,
                lr_edge_attr,
                ar_edge_attr,
                atom_edge_attr,
                la_edge_attr,
                ar_edge_attr,
            ],
            dim=0,
        )
        edge_sh = torch.cat(
            [
                lig_edge_sh,
                lr_edge_sh,
                la_edge_sh,
                rec_edge_sh,
                lr_edge_sh,
                ar_edge_sh,
                atom_edge_sh,
                la_edge_sh,
                ar_edge_sh,
            ],
            dim=0,
        )
        edge_weight = (
            torch.cat(
                [
                    lig_edge_weight,
                    lr_edge_weight,
                    la_edge_weight,
                    rec_edge_weight,
                    lr_edge_weight,
                    ar_edge_weight,
                    atom_edge_weight,
                    la_edge_weight,
                    ar_edge_weight,
                ],
                dim=0,
            )
            if torch.is_tensor(lig_edge_weight)
            else torch.ones((len(edge_index[0]), 1), device=edge_index.device)
        )
        s1, s2, s3, s4, s5, s6, s7, s8, _ = tuple(
            np.cumsum(
                list(
                    map(
                        len,
                        [
                            lig_edge_attr,
                            lr_edge_attr,
                            la_edge_attr,
                            rec_edge_attr,
                            lr_edge_attr,
                            ar_edge_attr,
                            atom_edge_attr,
                            la_edge_attr,
                            ar_edge_attr,
                        ],
                    )
                )
            ).tolist()
        )

        for layer in range(len(self.conv_layers)):
            edge_attr_ = torch.cat(
                [
                    edge_attr,
                    node_attr[edge_index[0], : self.ns],
                    node_attr[edge_index[1], : self.ns],
                ],
                -1,
            )
            if self.differentiate_convolutions:
                edge_attr_ = [
                    edge_attr_[:s1],
                    edge_attr_[s1:s2],
                    edge_attr_[s2:s3],
                    edge_attr_[s3:s4],
                    edge_attr_[s4:s5],
                    edge_attr_[s5:s6],
                    edge_attr_[s6:s7],
                    edge_attr_[s7:s8],
                    edge_attr_[s8:],
                ]
            node_attr = self.conv_layers[layer](
                node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight
            )

        lig_node_attr = node_attr[:n_lig]
        rec_node_attr = node_attr[n_lig : n_lig + n_rec]
        atom_node_attr = node_attr[n_lig + n_rec :]

        # confidence and affinity prediction
        if self.confidence_mode:
            scalar_lig_attr = (
                torch.cat(
                    [lig_node_attr[:, : self.ns], lig_node_attr[:, -self.ns :]], dim=1
                )
                if self.num_conv_layers >= 3
                else lig_node_attr[:, : self.ns]
            )
            scalar_lig_attr = scatter_mean(
                scalar_lig_attr,
                data["ligand"].batch,
                dim=0,
            )

            confidence_input = scalar_lig_attr

            if self.new_confidence_version:
                scalar_rec_attr = (
                    torch.cat(
                        [rec_node_attr[:, : self.ns], rec_node_attr[:, -self.ns :]],
                        dim=1,
                    )
                    if self.num_conv_layers >= 3
                    else rec_node_attr[:, : self.ns]
                )
                scalar_rec_attr = scatter_mean(
                    scalar_rec_attr,
                    data["receptor"].batch,
                    dim=0,
                )

                scalar_atom_attr = (
                    torch.cat(
                        [atom_node_attr[:, : self.ns], atom_node_attr[:, -self.ns :]],
                        dim=1,
                    )
                    if self.num_conv_layers >= 3
                    else atom_node_attr[:, : self.ns]
                )
                scalar_atom_attr = scatter_mean(
                    scalar_atom_attr,
                    data["atom"].batch,
                    dim=0,
                )

                confidence_input = torch.cat(
                    [scalar_lig_attr, scalar_rec_attr, scalar_atom_attr], dim=1
                )

            if self.atom_lig_confidence:
                atom_confidence_input = (
                    torch.cat(
                        [atom_node_attr[:, : self.ns], atom_node_attr[:, -self.ns :]],
                        dim=1,
                    )
                    if self.num_conv_layers >= 3
                    else atom_node_attr[:, : self.ns]
                )
                atom_confidence_input = scatter_mean(
                    atom_confidence_input,
                    data["atom"].batch,
                    dim=0,
                )
                atom_confidence = self.atom_confidence_predictor(
                    atom_confidence_input
                ).squeeze(dim=-1)

            confidence = self.confidence_predictor(confidence_input).squeeze(dim=-1)
            outputs = {"filtering_pred": confidence}
            if self.atom_lig_confidence:
                outputs["filtering_atom_pred"] = atom_confidence

            return outputs

        # compute translational and rotational score vectors
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[1], : self.ns]], -1
            )
        else:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[0], : self.ns]], -1
            )

        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=data.num_graphs,
        )

        tr_pred = global_pred[:, :3] + (
            global_pred[:, 6:9] if not self.odd_parity else 0
        )
        rot_pred = global_pred[:, 3:6] + (
            global_pred[:, 9:] if not self.odd_parity else 0
        )

        if self.separate_noise_schedule:
            data.graph_sigma_emb = torch.cat(
                [
                    self.timestep_emb_func(data.complex_t[noise_type])
                    for noise_type in ["tr", "rot", "tor", "sc_tor"]
                ],
                dim=1,
            )
        elif self.asyncronous_noise_schedule:
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t["t"])
        else:  # tr rot and tor noise is all the same in this case
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t["tr"])

        # adjust the magniture of the score vectors
        tr_norm = clamped_norm(tr_pred, dim=1, min=self.clamped_norm_min).unsqueeze(1)
        tr_pred = (
            tr_pred
            / tr_norm
            * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        )

        rot_norm = clamped_norm(rot_pred, dim=1, min=self.clamped_norm_min).unsqueeze(1)
        rot_pred = (
            rot_pred
            / rot_norm
            * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
        )

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(
                data["ligand"].x.device
            )

        if self.no_torsion or data["ligand"].edge_mask.sum() == 0:
            tor_pred = torch.empty(0, device=tr_pred.device)
        else:
            # torsional components
            (
                tor_bonds,
                tor_edge_index,
                tor_edge_attr,
                tor_edge_sh,
                tor_edge_weight,
            ) = self.build_bond_conv_graph(data)

            tor_bond_vec = (
                data["ligand"].pos[tor_bonds[1]] - data["ligand"].pos[tor_bonds[0]]
            )
            tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

            tor_bonds_sh = o3.spherical_harmonics(
                "2e", tor_bond_vec, normalize=True, normalization="component"
            )
            tor_edge_sh = self.final_tp_tor(
                tor_edge_sh, tor_bonds_sh[tor_edge_index[0]]
            )

            tor_edge_attr = torch.cat(
                [
                    tor_edge_attr,
                    lig_node_attr[tor_edge_index[1], : self.ns],
                    tor_bond_attr[tor_edge_index[0], : self.ns],
                ],
                -1,
            )

            tor_pred = self.tor_bond_conv(
                lig_node_attr,
                tor_edge_index,
                tor_edge_attr,
                tor_edge_sh,
                out_nodes=data["ligand"].edge_mask.sum(),
                reduce="mean",
                edge_weight=tor_edge_weight,
            )
            tor_pred = self.tor_final_layer(tor_pred).squeeze(1)

            if self.scale_by_sigma:
                edge_sigma = tor_sigma[data["ligand"].batch][
                    data["ligand", "lig_bond", "ligand"].edge_index[0]
                ][data["ligand"].edge_mask]
                tor_pred = tor_pred * torch.sqrt(
                    torch.tensor(torus.score_norm(edge_sigma.cpu().numpy()))
                    .float()
                    .to(data["ligand"].x.device)
                )

        if not self.flexible_sidechains:
            num_flexible_bonds = 0
        else:
            if fast_updates:
                num_flexible_bonds = data["atom", "atom_bond", "atom"].edge_mask.sum()
            else:
                if len(data["flexResidues"]) == 0:
                    num_flexible_bonds = 0
                else:
                    num_flexible_bonds = data["flexResidues"].edge_idx.shape[0]

        if num_flexible_bonds == 0:
            sc_tor_pred = torch.empty(0, device=tr_pred.device)
        else:
            try:
                # Torsion in sidechains (sc)
                (
                    sc_tor_bonds,
                    sc_tor_edge_index,
                    sc_tor_edge_attr,
                    sc_tor_edge_sh,
                    sc_tor_edge_weight,
                ) = self.build_sidechain_conv_graph(data, fast_updates=fast_updates)
                sc_tor_bond_vec = (
                    data["atom"].pos[sc_tor_bonds[1]]
                    - data["atom"].pos[sc_tor_bonds[0]]
                )
                sc_tor_bond_attr = (
                    atom_node_attr[sc_tor_bonds[0]] + atom_node_attr[sc_tor_bonds[1]]
                )

                sc_tor_bonds_sh = o3.spherical_harmonics(
                    "2e", sc_tor_bond_vec, normalize=True, normalization="component"
                )
                sc_tor_edge_sh = self.final_tp_sc_tor(
                    sc_tor_edge_sh, sc_tor_bonds_sh[sc_tor_edge_index[0]]
                )

                sc_tor_edge_attr = torch.cat(
                    [
                        sc_tor_edge_attr,
                        atom_node_attr[sc_tor_edge_index[1], : self.ns],
                        sc_tor_bond_attr[sc_tor_edge_index[0], : self.ns],
                    ],
                    -1,
                )

                sc_tor_pred = self.sc_tor_bond_conv(
                    atom_node_attr,
                    sc_tor_edge_index,
                    sc_tor_edge_attr,
                    sc_tor_edge_sh,
                    out_nodes=num_flexible_bonds,
                    reduce="mean",
                    edge_weight=sc_tor_edge_weight,
                )
                sc_tor_pred = self.sc_tor_final_layer(sc_tor_pred).squeeze(1)

                if self.scale_by_sigma and not self.sidechain_tor_bridge:
                    edge_sigma = sidechain_tor_sigma[
                        data["atom"].batch[
                            data["atom", "atom_bond", "atom"].edge_index[
                                0, data["atom", "atom_bond", "atom"].edge_mask
                            ]
                        ]
                    ]
                    norm = torch.sqrt(
                        torch.tensor(torus.score_norm(edge_sigma.cpu().numpy()))
                        .float()
                        .to(data["atom"].x.device)
                    )
                    sc_tor_pred = sc_tor_pred * norm

            except Exception as e:
                logging.error(
                    f"Exception encountered while predicting flexible sidechains for {data['name']}",
                )
                logging.error(e)
                logging.error(traceback.format_exc())
                raise e

        if self.flexible_backbone:
            # backbone drift prediction
            bb_pred = self.bb_o3_linear(rec_node_attr)
            bb_tr_pred = bb_pred[:, :3] + bb_pred[:, 6:9]
            bb_rot_pred = bb_pred[:, 3:6] + bb_pred[:, 9:]

            bb_tr_norm = clamped_norm(bb_tr_pred, dim=1, min=1e-6).unsqueeze(1)
            bb_tr_pred = (
                bb_tr_pred
                / bb_tr_norm
                * self.bb_tr_final_layer(
                    torch.cat([bb_tr_norm, data["receptor"].node_sigma_emb], dim=1)
                )
            )

            bb_rot_norm = clamped_norm(bb_rot_pred, dim=1, min=1e-6).unsqueeze(1)
            bb_rot_pred = (
                bb_rot_pred
                / bb_rot_norm
                * self.bb_rot_final_layer(
                    torch.cat([bb_rot_norm, data["receptor"].node_sigma_emb], dim=1)
                )
            )
        else:
            bb_tr_pred = bb_rot_pred = torch.empty(0, device=tr_pred.device)

        outputs = {
            "tr_pred": tr_pred,
            "rot_pred": rot_pred,
            "tor_pred": tor_pred,
            "bb_tr_pred": bb_tr_pred,
            "bb_rot_pred": bb_rot_pred,
            "sc_tor_pred": sc_tor_pred,
        }
        return outputs

    def get_edge_weight(self, edge_vec, max_norm):
        if self.smooth_edges:
            normalised_norm = torch.clip(
                edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi
            )
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    def build_lig_conv_graph(self, data):
        if self.separate_noise_schedule:
            data["ligand"].node_sigma_emb = torch.cat(
                [
                    self.timestep_emb_func(data["ligand"].node_t[noise_type])
                    for noise_type in ["tr", "rot", "tor", "sc_tor"]
                ],
                dim=1,
            )
        elif self.asyncronous_noise_schedule:
            data["ligand"].node_sigma_emb = self.timestep_emb_func(
                data["ligand"].node_t["t"]
            )
        else:
            data["ligand"].node_sigma_emb = self.timestep_emb_func(
                data["ligand"].node_t["tr"]
            )  # tr rot and tor noise is all the same

        radius_edges = radius_graph(
            data["ligand"].pos, self.lig_max_radius, data["ligand"].batch
        )
        edge_index = torch.cat(
            [data["ligand", "lig_bond", "ligand"].edge_index, radius_edges], 1
        ).long()
        edge_attr = torch.cat(
            [  # TODO: Be careful here, since self.in_lig_edge_features was set to 4 previously
                # and is now 5 for plinder
                data["ligand", "lig_bond", "ligand"].edge_attr[
                    :, : self.in_lig_edge_features
                ],
                torch.zeros(
                    radius_edges.shape[-1],
                    self.in_lig_edge_features,
                    device=data["ligand"].x.device,
                ),
            ],
            0,
        )

        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data):
        # build the graph between receptor residues
        if self.separate_noise_schedule:
            data["receptor"].node_sigma_emb = torch.cat(
                [
                    self.timestep_emb_func(data["receptor"].node_t[noise_type])
                    for noise_type in ["tr", "rot", "tor"]
                ],
                dim=1,
            )
        elif self.asyncronous_noise_schedule:
            data["receptor"].node_sigma_emb = self.timestep_emb_func(
                data["receptor"].node_t["t"]
            )
        else:
            data["receptor"].node_sigma_emb = self.timestep_emb_func(
                data["receptor"].node_t["tr"]
            )  # tr rot and tor noise is all the same
        node_attr = torch.cat([data["receptor"].x, data["receptor"].node_sigma_emb], 1)

        # construct knn graph receptor
        edge_index = knn_graph(
            data["receptor"].pos,
            k=self.c_alpha_max_neighbors if self.c_alpha_max_neighbors else 32,
            batch=data["receptor"].batch,
        )
        edge_vec = (
            data["receptor"].pos[edge_index[1].long()]
            - data["receptor"].pos[edge_index[0].long()]
        )
        edge_d = edge_vec.norm(dim=-1)
        if self.c_alpha_radius:
            to_keep = edge_d < self.c_alpha_radius
            edge_index = edge_index[:, to_keep]
            edge_vec = edge_vec[to_keep]
            edge_d = edge_d[to_keep]

        edge_length_emb = self.rec_distance_expansion(edge_d)
        edge_sigma_emb = data["receptor"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        if self.separate_noise_schedule:
            data["atom"].node_sigma_emb = torch.cat(
                [
                    self.timestep_emb_func(data["atom"].node_t[noise_type])
                    for noise_type in ["tr", "rot", "tor"]
                ],
                dim=1,
            )
        elif self.asyncronous_noise_schedule:
            data["atom"].node_sigma_emb = self.timestep_emb_func(
                data["atom"].node_t["t"]
            )
        else:
            data["atom"].node_sigma_emb = self.timestep_emb_func(
                data["atom"].node_t["tr"]
            )  # tr rot and tor noise is all the same
        node_attr = torch.cat([data["atom"].x, data["atom"].node_sigma_emb], 1)

        # construct knn graph atoms
        edge_index = knn_graph(
            data["atom"].pos,
            k=self.atom_max_neighbors if self.atom_max_neighbors else 32,
            batch=data["atom"].batch,
        )
        edge_vec = (
            data["atom"].pos[edge_index[1].long()]
            - data["atom"].pos[edge_index[0].long()]
        )
        edge_d = edge_vec.norm(dim=-1)
        if self.atom_radius:
            to_keep = edge_d < self.atom_radius
            edge_index = edge_index[:, to_keep]
            edge_vec = edge_vec[to_keep]
            edge_d = edge_d[to_keep]

        # data['atom', 'atom'].edge_index = edge_index
        edge_length_emb = self.lig_distance_expansion(edge_d)
        edge_sigma_emb = data["atom"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_index = radius(
                data["receptor"].pos / lr_cross_distance_cutoff[data["receptor"].batch],
                data["ligand"].pos / lr_cross_distance_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            lr_edge_index = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                lr_cross_distance_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )

        lr_edge_vec = (
            data["receptor"].pos[lr_edge_index[1].long()]
            - data["ligand"].pos[lr_edge_index[0].long()]
        )
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data["ligand"].node_sigma_emb[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, lr_edge_vec, normalize=True, normalization="component"
        )

        cutoff_d = (
            lr_cross_distance_cutoff[data["ligand"].batch[lr_edge_index[0]]].squeeze()
            if torch.is_tensor(lr_cross_distance_cutoff)
            else lr_cross_distance_cutoff
        )
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        # LIGAND to ATOM
        la_edge_index = radius(
            data["atom"].pos,
            data["ligand"].pos,
            self.lig_max_radius,
            data["atom"].batch,
            data["ligand"].batch,
            max_num_neighbors=10000,
        )

        la_edge_vec = (
            data["atom"].pos[la_edge_index[1].long()]
            - data["ligand"].pos[la_edge_index[0].long()]
        )
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data["ligand"].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, la_edge_vec, normalize=True, normalization="component"
        )
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        # ATOM to RECEPTOR
        ar_edge_index = data["atom", "receptor"].edge_index
        ar_edge_vec = (
            data["receptor"].pos[ar_edge_index[1].long()]
            - data["atom"].pos[ar_edge_index[0].long()]
        )
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data["atom"].node_sigma_emb[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, ar_edge_vec, normalize=True, normalization="component"
        )
        ar_edge_weight = 1

        return (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            lr_edge_weight,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            la_edge_weight,
            ar_edge_index,
            ar_edge_attr,
            ar_edge_sh,
            ar_edge_weight,
        )

    def build_center_conv_graph(self, data):
        # build the filter for the convolution of the center with the ligand atoms
        # for translational and rotational score
        edge_index = torch.cat(
            [
                data["ligand"].batch.unsqueeze(0),
                torch.arange(len(data["ligand"].batch))
                .to(data["ligand"].x.device)
                .unsqueeze(0),
            ],
            dim=0,
        )

        center_pos, _ = torch.zeros((data.num_graphs, 3)).to(
            data["ligand"].x.device
        ), torch.zeros((data.num_graphs, 3)).to(data["ligand"].x.device)
        center_pos.index_add_(0, index=data["ligand"].batch, source=data["ligand"].pos)
        center_pos = center_pos / torch.bincount(data["ligand"].batch).unsqueeze(1)

        edge_vec = data["ligand"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # build graph for the pseudotorque layer
        bonds = (
            data["ligand", "lig_bond", "ligand"]
            .edge_index[:, data["ligand"].edge_mask]
            .long()
        )
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        # determine for each bond the ligand atoms that lie within a certain distance
        edge_index = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
        )

        edge_vec = data["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return bonds, edge_index, edge_attr, edge_sh, edge_weight

    def build_sidechain_conv_graph(self, data, fast_updates: bool = False):
        # build graph for the pseudotorque layer
        if fast_updates:
            edge_mask_rotatable = data["atom", "atom_bond", "atom"].edge_mask
            bonds = data["atom", "atom_bond", "atom"].edge_index[:, edge_mask_rotatable]
            bond_batch = data["atom"].orig_batch[bonds[0]]
        else:
            bonds = TensorProductScoreModel.get_sc_tor_bonds(data)
            bond_batch = data[
                "flexResidues"
            ].batch  # TODO: is this correct? data['atom'].batch[bonds[0]]

        # The bonds / edge indices need to be remapped, since the pos, x and other attributes have been unmasked accordingly
        if self.only_nearby_residues_atomic:
            bonds = data["atom"].atom_new_idx_map[bonds]
            assert torch.all(
                bonds < 1000000
            ), "something went wrong with the filtering of nearby atoms"

        # assume that each bond lies between the two atoms that are connected
        bond_pos = (data["atom"].pos[bonds[0]] + data["atom"].pos[bonds[1]]) / 2
        # TODO: should we calculate the batch with radius? or use subcomponents
        # TODO: if we keep it this way, change self.lig_max_radius
        edge_index = radius(
            data["atom"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["atom"].batch,
            batch_y=bond_batch,
        )

        edge_vec = data["atom"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        # TODO: what to use here?
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.sidechain_final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        # TODO: what to use for edge weight?
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return bonds, edge_index, edge_attr, edge_sh, edge_weight

    @staticmethod
    def get_sc_tor_bonds(data):
        """
        This function returns the indices for data["atom"] to get the atoms of flexible sidechains
        The flexResidues are always indexed from 0...atoms but in a batched-scenario, this needs to be adjusted
        thus we add the correct offset in the ['atom'] graph
        """
        # We do not use bin counts because here we can be certain that the batches are sorted
        _, atom_bin_counts = data["atom"].orig_batch.unique(
            sorted=True, return_counts=True
        )

        bond_offset = atom_bin_counts.cumsum(dim=0)
        # shift it by one so to speak, because the first batch does not have an offset
        bond_offset = (
            torch.cat((torch.zeros(1, device=bond_offset.device), bond_offset))[:-1]
        ).long()
        # store the bonds of the flexible residues. i.e., we store which atoms are connected
        return (
            bond_offset[data["flexResidues"].batch]
            + data["flexResidues"].edge_idx.T.long()
        )
