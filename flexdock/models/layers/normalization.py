"""Based on the e3nn BatchNorm code"""
from typing import Optional, Literal

import torch
from torch import nn

from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.util.jit import compile_mode
from lightning.pytorch.utilities import rank_zero_info


@compile_mode("unsupported")
class LayerNorm(nn.Module):
    """Layer normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    include_bias : bool
        include a bias term for batch norm of scalars

    normalization : str
        which normalization method to apply (i.e., `norm` or `component`)
    """

    __constants__ = ["instance", "normalization", "irs", "affine"]

    def __init__(
        self,
        irreps: o3.Irreps,
        eps: float = 1e-5,
        affine: bool = True,
        reduce: str = "mean",
        include_bias: bool = True,
        normalization: str = "component",
    ) -> None:
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.include_bias = include_bias

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps
        self.features = []

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            if self.include_bias:
                if num_scalar:
                    self.bias = nn.Parameter(torch.zeros(num_scalar))
                else:
                    self.register_buffer("bias", torch.FloatTensor(0))
        else:
            self.register_parameter("weight", None)
            if self.include_bias:
                self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce
        irs = []
        for mul, ir in self.irreps:
            irs.append((mul, ir.dim, ir.is_scalar()))
        self.irs = irs

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, input) -> torch.Tensor:
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        orig_shape = input.shape
        batch = input.shape[0]
        dim = input.shape[-1]
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, d, is_scalar in self.irs:
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field: torch.Tensor = field.reshape(batch, -1, mul, d)

            if is_scalar:
                # mean along samples and multiplicity
                field_mean = torch.mean(field, (1, 2)).reshape(batch, 1)  # [batch, 1]
                field_mean = field_mean.expand(batch, mul)  # [batch, mul]

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
            else:
                raise ValueError(f"Invalid normalization option {self.normalization}")

            if self.reduce == "mean":
                field_norm = torch.mean(field_norm, (1, 2)).reshape(
                    batch, 1
                )  # [batch, 1]
            elif self.reduce == "max":
                field_norm = (
                    field_norm.max(1).values.max(1).values.reshape(batch, 1)
                )  # [batch, 1]
            else:
                raise ValueError(f"Invalid reduce option {self.reduce}")

            field_norm = field_norm.expand(batch, mul)  # [batch, mul]

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(
                -1, 1, mul, 1
            )  # [batch, sample, mul, repr]

            if self.affine and self.include_bias and is_scalar:
                bias = self.bias[ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(
                field.reshape(batch, -1, mul * d)
            )  # [batch, sample, mul * repr]

        torch._assert(
            ix == dim,
            f"`ix` should have reached input.size(-1) ({dim}), but it ended at {ix}",
        )

        if self.affine:
            torch._assert(iw == self.weight.size(0), "iw == self.weight.size(0)")
            if self.include_bias:
                torch._assert(ib == self.bias.numel(), "ib == self.bias.numel()")

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(orig_shape)


def get_norm_layer(
    out_irreps: o3.Irreps,
    norm_type: Optional[Literal["batch_norm", "layer_norm"]] = None,
    affine: bool = False,
) -> nn.Module:
    # check if norm_type is valid
    assert norm_type in [
        None,
        "batch_norm",
        "layer_norm",
    ], f"Invalid normalization type: {norm_type}"

    if norm_type == "batch_norm":
        norm = BatchNorm(out_irreps, affine=affine)
    elif norm_type == "layer_norm":
        norm = LayerNorm(out_irreps, affine=affine)
    else:
        rank_zero_info("No normalization layer would be used.")
        norm = nn.Identity()
    return norm
