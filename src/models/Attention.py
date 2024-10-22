from typing import Optional, Type
import torch
from torch import nn
from src.models.MLP import MLPStack

### Attention ###
class AttentionBlock(nn.Module):
    def __init__(
        self,
        output_size: int,
        depth: int,
        num_heads: int,
        norm: bool = True,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.projection_layer = nn.LazyLinear(output_size)
        # create some attention heads
        self.heads = nn.ModuleList(
            [
                MLPStack(
                    input_size=output_size,
                    hidden_size=output_size,
                    output_size=output_size,
                    depth=depth,
                    activation=activation,
                    norm=norm,
                    output_activation=activation,
                    residual=True,
                )
                for _ in range(num_heads)
            ]
        )
        self.attention = nn.Sequential(nn.LazyLinear(output_size), nn.Softmax())
        self.transform_layers = MLPStack(
            output_size, output_size, output_size, depth * 2, 0, activation, norm=norm, residual=False, lazy=True
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # project so we can use residual connections
        if data.ndim >= 3:
            data = data.view(data.shape[0], -1)
        projected_values = self.projection_layer(data)
        # stack up the results of each head
        outputs = torch.stack([head(projected_values) for head in self.heads], dim=1)
        weights = self.attention(outputs)
        weighted_values = (weights * outputs).flatten(1)
        return self.transform_layers(weighted_values)
