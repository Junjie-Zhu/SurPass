from typing import Union, Tuple, List

import torch
from torch import nn


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)