from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter

from lightning.pytorch.utilities import rank_zero_info

from flexdock.models.layers.normalization import get_norm_layer
from flexdock.models.layers.mlp import FCBlock


def get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars):
    if use_second_order_repr:
        irrep_seq = [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
            f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {nv if reduce_pseudoscalars else ns}x0o",
        ]
    else:
        irrep_seq = [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o",
            f"{ns}x0e + {nv}x1o + {nv}x1e",
            f"{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o",
        ]
    return irrep_seq


def get_position_vectors(irrep: str):
    irreps = irrep.split(" + ")
    size = 0
    pos_odd, pos_even = None, None
    for ir in irreps:
        m, (l, p) = ir.split("x")  # noqa: E741
        if l == "1":
            if p == "o":
                pos_odd = size
            else:
                pos_even = size
        size = size + int(m) * (2 * int(l) + 1)
    return pos_odd, pos_even


def irrep_to_size(irrep: str):
    irreps = irrep.split(" + ")
    size = 0
    for ir in irreps:
        m, (l, p) = ir.split("x")  # noqa: E741
        size += int(m) * (2 * int(l) + 1)
    return size


class FasterTensorProduct(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, **kwargs):
        super().__init__()
        # for ir in in_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order in irreps are not supported"
        # for ir in out_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order out irreps are not supported"
        assert o3.Irreps(sh_irreps) == o3.Irreps(
            "1x0e+1x1o"
        ), "sh_irreps don't look like 1st order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        in_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        out_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        for m, ir in self.in_irreps:
            in_muls[str(ir)] = m
        for m, ir in self.out_irreps:
            out_muls[str(ir)] = m

        self.weight_shapes = {
            "0e": (in_muls["0e"] + in_muls["1o"], out_muls["0e"]),
            "1o": (in_muls["0e"] + in_muls["1o"] + in_muls["1e"], out_muls["1o"]),
            "1e": (in_muls["1o"] + in_muls["1e"] + in_muls["0o"], out_muls["1e"]),
            "0o": (in_muls["1e"] + in_muls["0o"], out_muls["0o"]),
        }
        self.weight_numel = sum(a * b for (a, b) in self.weight_shapes.values())

    def forward(self, in_, sh, weight):
        in_dict, out_dict = {}, {"0e": [], "1o": [], "1e": [], "0o": []}
        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]
            if ir[0] == 1:
                in_dict[str(ir)] = in_dict[str(ir)].reshape(
                    list(in_dict[str(ir)].shape)[:-1] + [-1, 3]
                )
        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]
        if "0e" in in_dict:
            out_dict["0e"].append(in_dict["0e"] * sh_0e.unsqueeze(-1))
            out_dict["1o"].append(in_dict["0e"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if "1o" in in_dict:
            out_dict["0e"].append(
                (in_dict["1o"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3)
            )
            out_dict["1o"].append(in_dict["1o"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["1e"].append(
                torch.linalg.cross(in_dict["1o"], sh_1o.unsqueeze(-2), dim=-1)
                / np.sqrt(2)
            )
        if "1e" in in_dict:
            out_dict["1o"].append(
                torch.linalg.cross(in_dict["1e"], sh_1o.unsqueeze(-2), dim=-1)
                / np.sqrt(2)
            )
            out_dict["1e"].append(in_dict["1e"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["0o"].append(
                (in_dict["1e"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3)
            )
        if "0o" in in_dict:
            out_dict["1e"].append(in_dict["0o"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict["0o"].append(in_dict["0o"] * sh_0e.unsqueeze(-1))

        weight_dict = {}
        start = 0
        for key in self.weight_shapes:
            in_, out = self.weight_shapes[key]
            weight_dict[key] = weight[..., start : start + in_ * out].reshape(
                list(weight.shape)[:-1] + [in_, out]
            ) / np.sqrt(in_)
            start += in_ * out

        if out_dict["0e"]:
            out_dict["0e"] = torch.cat(out_dict["0e"], dim=-1)
            out_dict["0e"] = torch.matmul(
                out_dict["0e"].unsqueeze(-2), weight_dict["0e"]
            ).squeeze(-2)

        if out_dict["1o"]:
            out_dict["1o"] = torch.cat(out_dict["1o"], dim=-2)
            out_dict["1o"] = (
                out_dict["1o"].unsqueeze(-2) * weight_dict["1o"].unsqueeze(-1)
            ).sum(-3)
            out_dict["1o"] = out_dict["1o"].reshape(
                list(out_dict["1o"].shape)[:-2] + [-1]
            )

        if out_dict["1e"]:
            out_dict["1e"] = torch.cat(out_dict["1e"], dim=-2)
            out_dict["1e"] = (
                out_dict["1e"].unsqueeze(-2) * weight_dict["1e"].unsqueeze(-1)
            ).sum(-3)
            out_dict["1e"] = out_dict["1e"].reshape(
                list(out_dict["1e"].shape)[:-2] + [-1]
            )

        if out_dict["0o"]:
            out_dict["0o"] = torch.cat(out_dict["0o"], dim=-1)
            # out_dict['0o'] = (out_dict['0o'].unsqueeze(-1) * weight_dict['0o']).sum(-2)
            out_dict["0o"] = torch.matmul(
                out_dict["0o"].unsqueeze(-2), weight_dict["0o"]
            ).squeeze(-2)

        out = []
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)])
        return torch.cat(out, dim=-1)


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_irreps: str,
        sh_irreps: str,
        out_irreps: str,
        n_edge_features: int,
        residual: bool = True,
        dropout: float = 0.0,
        hidden_features: Optional[int] = None,
        faster: bool = False,
        edge_groups: int = 1,
        tp_weights_layers: int = 2,
        activation: str = "relu",
        depthwise: bool = False,
        norm_affine: bool = True,
        norm_type: Optional[Literal["batch_norm", "layer_norm"]] = None,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.edge_groups = edge_groups
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features

        if faster:
            rank_zero_info("Faster Tensor Product")
            self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
        else:
            self.tp = o3.FullyConnectedTensorProduct(
                in_irreps, sh_irreps, out_irreps, shared_weights=False
            )

        if edge_groups == 1:
            self.fc = FCBlock(
                n_edge_features,
                hidden_features,
                self.tp.weight_numel,
                tp_weights_layers,
                dropout,
                activation,
            )
        else:
            self.fc = [
                FCBlock(
                    n_edge_features,
                    hidden_features,
                    self.tp.weight_numel,
                    tp_weights_layers,
                    dropout,
                    activation,
                )
                for _ in range(edge_groups)
            ]
            self.fc = nn.ModuleList(self.fc)

        self.norm_type = norm_type
        self.norm = get_norm_layer(
            out_irreps=out_irreps, norm_type=norm_type, affine=norm_affine
        )

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_sh: torch.Tensor,
        out_nodes: int = None,
        reduce: str = "mean",
        edge_weight: float = 1.0,
    ):
        if edge_index.shape[1] == 0:
            # Ensure we get an output of desired shape if out_nodes is not None
            if out_nodes is None:
                out_nodes = node_attr.shape[0]
            out = torch.zeros(
                (out_nodes, self.out_size),
                dtype=node_attr.dtype,
                device=node_attr.device,
            )
        else:
            edge_src, edge_dst = edge_index
            edge_attr_ = (
                self.fc(edge_attr)
                if self.edge_groups == 1
                else torch.cat(
                    [self.fc[i](edge_attr[i]) for i in range(self.edge_groups)], dim=0
                ).to(node_attr.device)
            )

            # Apply Tensor Product
            tp = self.tp(node_attr[edge_dst], edge_sh, edge_attr_ * edge_weight)

            # Message Passing Operation
            out = scatter(
                tp,
                edge_src,
                dim=0,
                dim_size=node_attr.shape[0] if out_nodes is None else out_nodes,
                reduce=reduce,
            )

            # apply norm-operation
            out = self.norm(out)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        return out
