
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Callable


class CustomGLU(nn.Module):
    def __init__(self, activation: nn.Module, dim: int = -1):
        super(CustomGLU, self).__init__()
        self.dim = dim
        self.activation = activation

    def forward(self, x: Tensor):
        assert x.shape[self.dim] % 2 == 0          a, b = torch.chunk(x, 2, dim=self.dim)
        return a * self.activation(b)


class SwiGLU(CustomGLU):
    def __init__(self, dim: int = -1):
        super(SwiGLU, self).__init__(nn.SiLU(), dim)


class GeGLU(CustomGLU):
    def __init__(self, dim: int = -1):
        super(GeGLU, self).__init__(nn.GELU(), dim)


class ReGLU(CustomGLU):
    def __init__(self, dim: int = -1):
        super(ReGLU, self).__init__(nn.ReLU(), dim)


def get_activation_fn(
    activation: Union[str, Callable[[Tensor], Tensor]]
) -> Union[str, Callable[[Tensor], Tensor]]:
    if isinstance(activation, str):
        if activation == "reglu":
            return ReGLU()
        elif activation == "geglu":
            return GeGLU()
        elif activation == "swiglu":
            return SwiGLU()
    return activation
