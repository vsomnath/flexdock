from e3nn import o3
import torch
from typing import Optional, Literal

# from esm.pretrained import load_model_and_alphabet
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
import numpy as np


try:
    from relaxflow.utils.forces import get_bonded_forces, load_model_and_alphabet
except Exception:
    pass

from flexdock.data.constants import (
    lig_feature_dims,
    rec_residue_feature_dims,
    rec_atom_feature_dims,
)
from flexdock.models.layers.tensor_product import TensorProductConvLayer
from flexdock.models.networks.encoders import AtomEncoder, GaussianSmearing


def edge_index_to_edge_idx(edge_index, batch):
    edge_batch = batch[edge_index[0]]
    counts = torch.bincount(batch)
    ptr = torch.cat(
        [torch.tensor([0], device=counts.device), torch.cumsum(counts, 0)[:-1]]
    )
    edge_ptr = torch.cat(
        [
            torch.tensor([0], device=counts.device),
            torch.cumsum(counts * (counts - 1) // 2, 0)[:-1],
        ]
    )
    edge_index = edge_index.clone().sort(dim=0).values - ptr[edge_batch]
    edge_idx = (
        edge_ptr[edge_batch]
        + (edge_index[0] * (2 * counts[edge_batch] - edge_index[0] - 3) // 2)
        + edge_index[1]
        - 1
    )
    return edge_idx


def get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars, use_forces=True):
    if use_second_order_repr:
        irrep_seq = [
            f"{ns}x0e" + (" + 1x1o" if use_forces else ""),
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {nv if reduce_pseudoscalars else ns}x0o",
        ]
    else:
        irrep_seq = [
            f"{ns}x0e" + (" + 1x1o" if use_forces else ""),
            f"{ns}x0e + {nv}x1o",
            f"{ns}x0e + {nv}x1o + {nv}x1e",
            f"{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o",
        ]
    return irrep_seq


class TensorProductFlowModel(torch.nn.Module):
    def __init__(
        self,
        device,
        timestep_emb_func,
        in_lig_edge_features=3,
        lig_max_radius=5,
        lig_max_neighbors=None,
        rec_max_radius=30,
        rec_max_neighbors=32,
        atom_max_radius=5,
        atom_max_neighbors=None,
        cross_max_radius=250,
        cross_max_neighbors=None,
        sigma_embed_dim=32,
        distance_embed_dim=32,
        cross_distance_embed_dim=32,
        lm_embedding_type=None,
        num_prot_emb_layers=0,
        embed_also_ligand=False,
        num_conv_layers=2,
        ns=16,
        nv=4,
        sh_lmax=2,
        use_second_order_repr=False,
        reduce_pseudoscalars=False,
        differentiate_convolutions=True,
        tp_weights_layers=2,
        smooth_edges=False,
        batch_norm=True,
        dropout=0.0,
        use_forces=False,
        embed_radii=False,
        embed_bounds=False,
        norm_type: Optional[Literal["batch_norm", "layer_norm"]] = None,
        norm_affine: bool = True,
    ):
        self.bond_distance_embed_dim = 0  # 64
        super(TensorProductFlowModel, self).__init__()
        self.device = device
        self.in_lig_edge_features = 4  # in_lig_edge_features

        self.lig_max_radius = lig_max_radius
        self.lig_max_neighbors = lig_max_neighbors
        self.rec_max_radius = rec_max_radius
        self.rec_max_neighbors = rec_max_neighbors
        self.atom_max_radius = atom_max_radius
        self.atom_max_neighbors = atom_max_neighbors
        self.cross_max_radius = cross_max_radius
        self.cross_max_neighbors = cross_max_neighbors

        self.timestep_emb_func = timestep_emb_func
        self.sigma_embed_dim = sigma_embed_dim
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim

        self.num_prot_emb_layers = num_prot_emb_layers
        self.embed_also_ligand = embed_also_ligand
        self.num_conv_layers = num_conv_layers
        self.ns, self.nv = ns, nv

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.smooth_edges = smooth_edges
        self.differentiate_convolutions = differentiate_convolutions
        self.reduce_pseudoscalars = reduce_pseudoscalars

        self.embed_radii = embed_radii
        self.embed_bounds = embed_bounds
        self.lm_embedding_type = None  # lm_embedding_type
        lm_embedding_type = None
        self.use_forces = use_forces
        if lm_embedding_type is None:
            lm_embedding_dim = 0
        elif lm_embedding_type == "precomputed":
            lm_embedding_dim = 1280
        else:
            lm, alphabet = load_model_and_alphabet(lm_embedding_type)
            self.batch_converter = alphabet.get_batch_converter()
            lm.lm_head = torch.nn.Identity()
            lm.contact_head = torch.nn.Identity()
            lm_embedding_dim = lm.embed_dim  # noqa: F841
            self.lm = lm

        # embedding layers
        atom_encoder_class = AtomEncoder
        self.lig_node_embedding = atom_encoder_class(
            emb_dim=ns,
            feature_dims=lig_feature_dims,
            sigma_embed_dim=sigma_embed_dim
            + (distance_embed_dim if self.embed_radii else 0),
        )
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(
                self.in_lig_edge_features
                + sigma_embed_dim
                + distance_embed_dim * (3 if self.embed_bounds else 1),
                ns,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.rec_node_embedding = atom_encoder_class(
            emb_dim=ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=sigma_embed_dim,
            lm_embedding_type=None
            # lm_embedding_type="esm",
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.atom_node_embedding = atom_encoder_class(
            emb_dim=ns,
            feature_dims=rec_atom_feature_dims,
            sigma_embed_dim=sigma_embed_dim
            + (distance_embed_dim if self.embed_radii else 0),
        )
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            nn.ReLU(),
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
            0.0, cross_max_radius, cross_distance_embed_dim
        )
        self.vdw_distance_expansion = GaussianSmearing(1.0, 3.0, distance_embed_dim)
        irrep_seq = get_irrep_seq(
            ns,
            nv,
            use_second_order_repr,
            reduce_pseudoscalars,
            use_forces=self.use_forces,
        )

        rec_emb_layers = []
        for i in range(num_prot_emb_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                norm_type=norm_type,
                norm_affine=norm_affine,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else 4,
            )
            rec_emb_layers.append(layer)
        self.rec_emb_layers = nn.ModuleList(rec_emb_layers)

        if embed_also_ligand:
            lig_emb_layers = []
            for i in range(num_prot_emb_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
                layer = TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=True,
                    norm_type=norm_type,
                    norm_affine=norm_affine,
                    dropout=dropout,
                    faster=sh_lmax == 1 and not use_second_order_repr,
                    tp_weights_layers=tp_weights_layers,
                    edge_groups=1,
                )
                lig_emb_layers.append(layer)
            self.lig_emb_layers = nn.ModuleList(lig_emb_layers)

        # convolutional layers
        conv_layers = []
        for i in range(num_prot_emb_layers, num_prot_emb_layers + num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                norm_type=norm_type,
                norm_affine=norm_affine,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else 9
                # (3 if i == num_prot_emb_layers + num_conv_layers - 1 else 9),
            )
            conv_layers.append(layer)
        final_conv_layer = TensorProductConvLayer(
            in_irreps=conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps="1x1o",
            n_edge_features=3 * ns,
            hidden_features=3 * ns,
            residual=True,
            norm_type=norm_type,
            norm_affine=norm_affine,
            dropout=dropout,
            faster=sh_lmax == 1 and not use_second_order_repr,
            tp_weights_layers=tp_weights_layers,
            edge_groups=1 if not differentiate_convolutions else 9,
        )
        conv_layers.append(final_conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.final_layer = nn.Sequential(
            nn.Linear(1 + sigma_embed_dim, ns),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ns, 1),
        )

    def embedding(self, data):
        if self.lm_embedding_type not in [None, "precomputed"]:
            sequences = [s for l in data["receptor"].sequence for s in l]
            if isinstance(sequences[0], list):
                sequences = [s for l in sequences for s in l]
            sequences = [(i, s) for i, s in enumerate(sequences)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
            out = self.lm(
                batch_tokens.to(data["receptor"].x.device),
                repr_layers=[self.lm.num_layers],
                return_contacts=False,
            )
            rec_lm_emb = torch.cat(
                [
                    t[: len(sequences[i][1])]
                    for i, t in enumerate(out["representations"][self.lm.num_layers])
                ],
                dim=0,
            )
            data["receptor"].x = torch.cat([data["receptor"].x, rec_lm_emb], dim=-1)

        # import pdb; pdb.set_trace()
        (
            rec_node_attr,
            rec_edge_attr,
            rec_edge_sh,
            rec_edge_weight,
        ) = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        (
            atom_node_attr,
            atom_edge_attr,
            atom_edge_sh,
            atom_edge_weight,
        ) = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

        ar_edge_attr, ar_edge_sh, ar_edge_weight = self.build_cross_rec_conv_graph(data)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

        rec_edge_index = data["receptor", "receptor"].edge_index.clone()
        atom_edge_index = data["atom", "atom"].edge_index.clone()
        ar_edge_index = data["atom", "receptor"].edge_index.clone()

        if self.use_forces:
            lig_force, atom_force = get_bonded_forces(data)
            lig_force = lig_force / torch.linalg.norm(lig_force, axis=-1, keepdims=True)
            atom_force = atom_force / torch.linalg.norm(
                atom_force, axis=-1, keepdims=True
            )

            atom_node_attr = torch.cat((atom_node_attr, atom_force), dim=1)
            rec_node_attr = torch.cat(
                (
                    rec_node_attr,
                    torch.zeros(
                        (rec_node_attr.shape[0], 3), device=rec_node_attr.device
                    ),
                ),
                dim=1,
            )

        node_attr = torch.cat([rec_node_attr, atom_node_attr], dim=0)
        ar_edge_index[0] = ar_edge_index[0] + len(rec_node_attr)
        edge_index = torch.cat(
            [
                rec_edge_index,
                ar_edge_index,
                atom_edge_index + len(rec_node_attr),
                torch.flip(ar_edge_index, dims=[0]),
            ],
            dim=1,
        )
        edge_attr = torch.cat(
            [rec_edge_attr, ar_edge_attr, atom_edge_attr, ar_edge_attr], dim=0
        )
        edge_sh = torch.cat([rec_edge_sh, ar_edge_sh, atom_edge_sh, ar_edge_sh], dim=0)
        edge_weight = (
            torch.cat(
                [rec_edge_weight, ar_edge_weight, atom_edge_weight, ar_edge_weight],
                dim=0,
            )
            if torch.is_tensor(rec_edge_weight)
            else torch.ones((len(edge_index[0]), 1), device=edge_index.device)
        )
        s1, s2, s3 = (
            len(rec_edge_index[0]),
            len(rec_edge_index[0]) + len(ar_edge_index[0]),
            len(rec_edge_index[0]) + len(ar_edge_index[0]) + len(atom_edge_index[0]),
        )

        for l in range(len(self.rec_emb_layers)):
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
                    edge_attr_[s3:],
                ]
            node_attr = self.rec_emb_layers[l](
                node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight
            )

        data["receptor"].rec_node_attr = node_attr[: len(rec_node_attr)]
        data["receptor", "receptor"].rec_edge_attr = rec_edge_attr
        data["receptor", "receptor"].edge_sh = rec_edge_sh
        data["receptor", "receptor"].edge_weight = rec_edge_weight

        data["atom"].atom_node_attr = node_attr[len(rec_node_attr) :]
        data["atom", "atom"].atom_edge_attr = atom_edge_attr
        data["atom", "atom"].edge_sh = atom_edge_sh
        data["atom", "atom"].edge_weight = atom_edge_weight

        data["atom", "receptor"].edge_attr = ar_edge_attr
        data["atom", "receptor"].edge_sh = ar_edge_sh
        data["atom", "receptor"].edge_weight = ar_edge_weight

        # ligand embedding
        (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
            lig_edge_weight,
        ) = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        if self.use_forces:
            lig_node_attr = torch.cat((lig_node_attr, lig_force), dim=1)
        if self.embed_also_ligand:
            for l in range(len(self.lig_emb_layers)):
                edge_attr_ = torch.cat(
                    [
                        lig_edge_attr,
                        lig_node_attr[lig_edge_index[0], : self.ns],
                        lig_node_attr[lig_edge_index[1], : self.ns],
                    ],
                    -1,
                )
                lig_node_attr = self.lig_emb_layers[l](
                    lig_node_attr,
                    lig_edge_index,
                    edge_attr_,
                    lig_edge_sh,
                    edge_weight=lig_edge_weight,
                )

        else:
            lig_node_attr = F.pad(
                lig_node_attr, (0, rec_node_attr.shape[-1] - lig_node_attr.shape[-1])
            )

        return (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
            lig_edge_weight,
            rec_node_attr,
            data["receptor", "receptor"].edge_index,
            rec_edge_attr,
            data["receptor", "receptor"].edge_sh,
            data["receptor", "receptor"].edge_weight,
            atom_node_attr,
            data["atom", "atom"].edge_index,
            atom_edge_attr,
            data["atom", "atom"].edge_sh,
            data["atom", "atom"].edge_weight,
            data["atom", "receptor"].edge_index,
            ar_edge_attr,
            data["atom", "receptor"].edge_sh,
            data["atom", "receptor"].edge_weight,
        )

    def forward(self, data):
        (
            lig_node_attr,
            lig_edge_index,
            lig_edge_attr,
            lig_edge_sh,
            lig_edge_weight,
            rec_node_attr,
            rec_edge_index,
            rec_edge_attr,
            rec_edge_sh,
            rec_edge_weight,
            atom_node_attr,
            atom_edge_index,
            atom_edge_attr,
            atom_edge_sh,
            atom_edge_weight,
            ar_edge_index,
            ar_edge_attr,
            ar_edge_sh,
            ar_edge_weight,
        ) = self.embedding(data)

        # build lig cross graph
        (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            lr_edge_weight,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            la_edge_weight,
        ) = self.build_cross_lig_conv_graph(data, self.cross_max_radius)
        lr_edge_attr = self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)

        n_lig, n_rec = len(lig_node_attr), len(rec_node_attr)

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

        for l in range(len(self.conv_layers)):
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
            node_attr = self.conv_layers[l](
                node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight
            )

        lig_node_attr = node_attr[: len(lig_node_attr)]
        atom_node_attr = node_attr[-1 * len(atom_node_attr) :]

        # final layers
        lig_norm = torch.linalg.vector_norm(lig_node_attr, dim=1).unsqueeze(1)
        lig_pred = (
            lig_node_attr
            / lig_norm
            * self.final_layer(
                torch.cat([lig_norm, data["ligand"].node_sigma_emb], dim=1)
            )
        )
        atom_norm = torch.linalg.vector_norm(atom_node_attr, dim=1).unsqueeze(1)
        atom_pred = (
            atom_node_attr
            / atom_norm
            * self.final_layer(
                torch.cat([atom_norm, data["atom"].node_sigma_emb], dim=1)
            )
        )

        return lig_pred, atom_pred

    def get_edge_weight(self, edge_vec, max_norm):
        if self.smooth_edges:
            normalised_norm = torch.clip(
                edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi
            )
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    def build_lig_conv_graph(self, data):
        # build the graph between ligand atoms
        data["ligand"].node_sigma_emb = self.timestep_emb_func(data["ligand"].node_t)

        radius_edges = radius_graph(
            data["ligand"].pos, self.lig_max_radius, data["ligand"].batch
        )
        edge_index = torch.cat(
            [data["ligand", "lig_bond", "ligand"].edge_index, radius_edges], 1
        ).long()
        edge_attr = torch.cat(
            [
                data["ligand", "lig_bond", "ligand"].edge_attr[
                    :, : self.in_lig_edge_features
                ],
                torch.zeros(
                    radius_edges.shape[-1],
                    self.in_lig_edge_features,
                    # data["ligand", "lig_bond", "ligand"].edge_attr.shape[1],
                    device=data["ligand"].x.device,
                ),
            ],
            0,
        )

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[src.long()]

        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], 1)
        edge_attr = torch.cat([edge_attr, edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        if self.embed_radii:
            vdw_attr = self.vdw_distance_expansion(data["ligand"].vdw_radii)
            node_attr = torch.cat([node_attr, vdw_attr], 1)
        if self.embed_bounds:
            full_edge_idx = edge_index_to_edge_idx(edge_index, data["ligand"].batch)
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self.lig_distance_expansion(
                        data["ligand", "lig_edge", "ligand"]
                        .lower_bound[full_edge_idx]
                        .float()
                    ),
                    self.lig_distance_expansion(
                        data["ligand", "lig_edge", "ligand"]
                        .upper_bound[full_edge_idx]
                        .float()
                    ),
                ),
                dim=1,
            )

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data):
        # build the graph between receptor residues
        data["receptor"].node_sigma_emb = self.timestep_emb_func(
            data["receptor"].node_t
        )

        data["receptor", "receptor"].edge_index = radius_graph(
            data["receptor"].pos,
            self.rec_max_radius,
            data["receptor"].batch,
            max_num_neighbors=data["receptor"].num_nodes
            if self.rec_max_neighbors is None
            else self.rec_max_neighbors,
        )

        src, dst = data["receptor", "receptor"].edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["receptor"].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["receptor"].node_sigma_emb[src.long()]

        node_attr = torch.cat([data["receptor"].x, data["receptor"].node_sigma_emb], 1)
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)

        return node_attr, edge_attr, edge_sh, edge_weight

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        data["atom"].node_sigma_emb = self.timestep_emb_func(data["atom"].node_t)

        radius_edges = radius_graph(
            data["atom"].pos,
            self.atom_max_radius,
            data["atom"].batch,
            max_num_neighbors=data["atom"].num_nodes
            if self.atom_max_neighbors is None
            else self.atom_max_neighbors,
        )

        edge_index = radius_edges
        data["atom", "atom"].edge_index = edge_index

        src, dst = edge_index
        edge_vec = data["atom"].pos[dst.long()] - data["atom"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["atom"].node_sigma_emb[src.long()]

        node_attr = torch.cat([data["atom"].x, data["atom"].node_sigma_emb], 1)
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        if self.embed_radii:
            vdw_attr = self.vdw_distance_expansion(data["atom"].vdw_radii)
            node_attr = torch.cat([node_attr, vdw_attr], 1)

        return node_attr, edge_attr, edge_sh, edge_weight

    def build_cross_lig_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligand atoms and receptor residues + atoms

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
        la_edge_length_emb = self.lig_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data["ligand"].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, la_edge_vec, normalize=True, normalization="component"
        )
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        return (
            lr_edge_index,
            lr_edge_attr,
            lr_edge_sh,
            lr_edge_weight,
            la_edge_index,
            la_edge_attr,
            la_edge_sh,
            la_edge_weight,
        )

    def build_cross_rec_conv_graph(self, data):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # ATOM to RECEPTOR
        ar_edge_index = data["atom", "receptor"].edge_index
        ar_edge_vec = (
            data["receptor"].pos[ar_edge_index[1].long()]
            - data["atom"].pos[ar_edge_index[0].long()]
        )
        ar_edge_attr = torch.cat(
            [
                data["atom"].node_sigma_emb[ar_edge_index[0].long()],
                self.rec_distance_expansion(ar_edge_vec.norm(dim=-1)),
            ],
            axis=1,
        )
        ar_edge_sh = o3.spherical_harmonics(
            self.sh_irreps, ar_edge_vec, normalize=True, normalization="component"
        )
        ar_edge_weight = 1

        return ar_edge_attr, ar_edge_sh, ar_edge_weight
