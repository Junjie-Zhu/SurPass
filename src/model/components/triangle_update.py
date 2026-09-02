from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _stable_triangle_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Contract over K in fp32 and scale by 1/sqrt(K) so LayerNorm does not see Inf."""
    k = int(a.shape[-1])
    out = torch.matmul(a.float(), b.float())
    if k > 0:
        out = out * (float(k) ** -0.5)
    return out


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden)
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden)
        self.linear_g = Linear(self.c_z, self.c_z)
        self.linear_z = Linear(self.c_hidden, self.c_z)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overridden")

    def _layer_norm_out(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.layer_norm_out.weight
        bias = self.layer_norm_out.bias
        return F.layer_norm(
            x.float(),
            self.layer_norm_out.normalized_shape,
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
            self.layer_norm_out.eps,
        )

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask
        x = self._layer_norm_out(self._combine_projections(a, b))
        x = self.linear_z(x.to(dtype=z.dtype))
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_i, N_k, C]
        b: torch.Tensor,  # [*, N_j, N_k, C]
    ):
        p = _stable_triangle_matmul(
            permute_final_dims(a, (2, 0, 1)),
            permute_final_dims(b, (2, 1, 0)),
        )
        return permute_final_dims(p, (1, 2, 0))


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    def _combine_projections(
        self,
        a: torch.Tensor,  # [*, N_k, N_i, C]
        b: torch.Tensor,  # [*, N_k, N_j, C]
    ):
        p = _stable_triangle_matmul(
            permute_final_dims(a, (2, 1, 0)),
            permute_final_dims(b, (2, 0, 1)),
        )
        return permute_final_dims(p, (1, 2, 0))
