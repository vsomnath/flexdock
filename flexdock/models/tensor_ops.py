# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import degree


def permute_final_dims(tensor, *inds):
    zero_index = -1 * len(inds)
    first_inds = range(len(tensor.shape[:zero_index]))
    return tensor.permute(*first_inds, *[zero_index + i for i in inds])


def flatten_final_dims(tensor, no_dims):
    return tensor.reshape(*tensor.shape[:-no_dims], -1)


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    tree_type = type(tree)
    if tree_type is dict:
        return dict_map(fn, tree, leaf_type)
    elif tree_type is list:
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif tree_type is tuple:
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif tree_type is leaf_type:
        return fn(tree)
    else:
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def chunk_layer(layer, inputs, chunk_size, no_batch_dims):
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are interpreted as simplified "pytrees,"
    consisting only of (nested) lists, tuples, and dicts with tensor
    leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (nested) dictionary of keyworded inputs. All leaves must be
            tensors and must share the same batch dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch
            dimensions are specified, a "sub-batch" is defined as a single
            indexing of all batch dimensions simultaneously (s.t. the
            number of sub-batches is the product of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can
            be considered batch dimensions.
    Returns:
        The reassembled output of the layer on the inputs.
    """

    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    def fetch_dims(tree):
        shapes = []
        tree_type = type(tree)
        if tree_type is dict:
            for v in tree.values():
                shapes.extend(fetch_dims(v))
        elif tree_type is list or tree_type is tuple:
            for t in tree:
                shapes.extend(fetch_dims(t))
        elif tree_type is torch.Tensor:
            shapes.append(tree.shape)
        else:
            raise ValueError("Not supported")

        return shapes

    initial_dims = [shape[:no_batch_dims] for shape in fetch_dims(inputs)]
    orig_batch_dims = [max(s) for s in zip(*initial_dims)]

    def prep_inputs(t):
        t = t.expand(*orig_batch_dims, *t.shape[no_batch_dims:])
        t = t.reshape(-1, *t.shape[no_batch_dims:])
        return t

    # shape = lambda t: t.shape
    # print(tensor_tree_map(shape, inputs))

    flattened_inputs = tensor_tree_map(prep_inputs, inputs)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    i = 0
    out = None
    for _ in range(no_chunks):
        # Chunk the input
        select_chunk = lambda t: t[i : i + chunk_size]
        chunks = tensor_tree_map(select_chunk, flattened_inputs)

        # Run the layer on the chunk
        output_chunk = layer(**chunks)

        # Allocate space for the output
        if out is None:
            allocate = lambda t: t.new_zeros(flat_batch_dim, *t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        # Put the chunk in its pre-allocated space
        out_type = type(output_chunk)
        if out_type is dict:

            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.reshape(*orig_batch_dims, *t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out


def unbatch(src, batch: Tensor, dim: int = 0) -> list[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`list[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    if isinstance(src, np.ndarray):
        return np.split(src, np.array(sizes).cumsum()[:-1], axis=dim)
    else:
        return src.split(sizes, dim)


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> list[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`list[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int32).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def unbatch_edge_attributes(
    edge_attributes, edge_index: Tensor, batch: Tensor
) -> list[Tensor]:
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attributes.split(sizes, dim=0)


def index_expand_and_select(input, dim, index):
    """
    Equivelant to doing input[:, :, ... , index, :, :, ... ] where input is indexed at dimension dim and index can be N dimensional
    """
    return torch.gather(
        input.view(
            input.shape[:dim]
            + tuple(1 for _ in range(len(index.shape[:-1])))
            + input.shape[dim:]
        ).expand(input.shape[:dim] + index.shape[:-1] + input.shape[dim:]),
        dim,
        index.view(
            tuple(1 for _ in range(len(input.shape[:dim])))
            + index.shape
            + tuple(1 for _ in range(len(input.shape[dim + 1 :])))
        ).expand(input.shape[:dim] + index.shape + input.shape[dim + 1 :]),
    )


def clamped_norm(vec, dim=1, min=1e-6):
    return torch.clamp(torch.linalg.vector_norm(vec, dim=dim), min=min)
