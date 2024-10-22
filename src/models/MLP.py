from typing import Optional, Type
import torch
from torch import nn


### MLP ###
class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = None,
        norm: bool = True,
        residual: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        linear = nn.LazyLinear(output_size) if lazy else nn.Linear(input_size, output_size)
        norm_layer = nn.LazyBatchNorm1d(output_size) if lazy and norm else nn.BatchNorm1d(output_size) if norm else nn.Identity()
        activation_layer = nn.Identity() if activation is None else activation()
        dropout_layer = nn.Dropout(dropout)
        self.model = nn.Sequential(linear, norm_layer, activation_layer, dropout_layer)
        self.residual = residual

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual:
            output = output + data
        return output


class MLPStack(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        output_activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm: bool = True,
        residual: bool = False,
        residual_full: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        blocks = [MLPBlock(input_size, hidden_size, dropout, activation, norm=norm, residual=False, lazy=lazy)]
        for _ in range(depth - 1):
            blocks.append(MLPBlock(hidden_size, hidden_size, dropout, activation, norm=norm, residual=residual, lazy=False))
        blocks.append(MLPBlock(hidden_size, output_size, dropout, output_activation, norm=norm, residual=False, lazy=False))
        self.model = nn.Sequential(*blocks)
        self.residual_full = residual_full
        self.norm = nn.BatchNorm1d(output_size) if norm else nn.Identity()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.model(data)
        if self.residual_full:
            output = output + data
        output = self.norm(output)
        return output