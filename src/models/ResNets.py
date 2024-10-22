from typing import Optional, Type
import numpy as np
import torch
from torch import nn
from src.models.Attention import AttentionBlock

### CNN / ResNet blocks ###
class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm=True,
        residual: bool = True,
    ) -> None:
        super(ResBlock1d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity(),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


class AttnResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm: bool = True,
        residual: bool = True,
        attn_on: bool = False,
        attn_depth: int = 1,
        attn_heads: int = 2,
    ) -> None:
        super().__init__()

        # Whether or not to activate ResNet block skip connections
        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=not norm)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding, bias=not norm)
        self.bn2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()

        # Add or skip attention layer based on the use_attention flag
        self.attention = AttentionBlock(out_channels, attn_depth, attn_heads, norm) if attn_on else nn.Identity()

        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        x = x[0] if isinstance(x, tuple) else x  # get only x, ignore residual that is fed back into forward pass
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.attention(residual)  # Apply attention if available

        if self.residual:  # forward skip connection
            out += residual*self.residual_scale

        out = self.activation(out)

        return out, residual


class AttnResBlock1dT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        upsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0,
        norm=True,
        residual: bool = True,
        attn_on: bool = False,
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.residual_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.activation = nn.Identity() if activation is None else activation()
        padding = kernel // 2

        self.convt1 = nn.ConvTranspose1d(
            in_channels, in_channels, kernel_size=kernel, stride=1, padding=padding, output_padding=0, bias=not norm
        )

        self.bn1 = nn.BatchNorm1d(in_channels) if norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else activation()

        # out_channels may be wrong argument here
        self.attention = AttentionBlock(out_channels, attn_depth, attn_heads, norm) if attn_on else nn.Identity()

        self.convt2 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride - 1, bias=not norm
        )

        self.bn2 = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.upsample = upsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.convt1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.attention(out)

        out = self.convt2(out)
        out = self.bn2(out)

        if self.upsample:
            residual = self.upsample(x)
        if self.residual:
            out += residual*self.residual_scale
        out = self.activation(out)
        return out


class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(3, 3),
        stride=1,
        downsample=None,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        dropout=0.0,
        norm=True,
        residual: bool = True,
    ) -> None:
        super(ResBlock2d, self).__init__()

        self.residual = residual
        self.activation = nn.Identity() if activation is None else activation()
        padding = tuple(k // 2 for k in kernel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                bias=not norm
            ),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # get only x, ignore residual that is fed back into forward pass
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        if self.residual:  # forward skip connection
            out += residual
        out = self.activation(out)
        return out, residual


### ResNets ###
class AttnResNet1d(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock1d,
        depth: int = 4,
        channels: list = [1, 64, 128, 256, 512],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        residual: bool = False,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # List of 0s and 1s indicating whether attention is applied in each layer
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.attn_on = attn_on

        self.layers = nn.ModuleDict({})
        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])

            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1],
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=not norm),
                nn.BatchNorm1d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                downsample,
                activation,
                dropout,
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        residuals = []
        for i in range(0, self.depth):
            x, res = self.layers[str(i)](x)
            residuals.append(res)

        return x, residuals


class AttnResNet1dT(nn.Module):
    def __init__(
        self,
        block: nn.Module = AttnResBlock1dT,
        depth: int = 4,
        channels: list = [512, 256, 128, 64, 1],
        kernels: list = [3, 3, 3, 3, 3],
        strides: list = [1, 1, 1, 1, 1],
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=True,
        sym_residual: bool = True,
        fwd_residual: bool = True,
        attn_on: list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_depth: int = 1,
        attn_heads: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.inplanes = channels[0]
        self.sym_residual = sym_residual  # for symmetric skip connections
        self.fwd_residual = fwd_residual  # for forward (normal) skip connections
        self.attn_on = attn_on
        self.residual_scales = nn.ParameterList([nn.Parameter(torch.tensor([1.0]), requires_grad=True) for _ in range(depth)])

        self.layers = nn.ModuleDict({})
        # self.fusion_layers = nn.ModuleDict({})

        for i in range(0, self.depth):
            attn_enabled = False if self.attn_on is None else bool(self.attn_on[i])
            self.layers[str(i)] = self._make_layer(
                block,
                channels[i + 1], # // 2, # CCCCC
                kernel=kernels[i],
                stride=strides[i],
                dropout=dropout,
                activation=activation,
                norm=norm,
                residual=fwd_residual,
                attn_on=attn_enabled,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )

            # self.fusion_layers[str(i)] = Conv2DFusion(channels[i])

    def _make_layer(
        self, block, planes, kernel, stride, dropout, activation, norm, residual, attn_on, attn_depth, attn_heads
    ):
        upsample = None
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose1d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    output_padding=stride - 1,
                    bias=not norm
                ),
                nn.BatchNorm1d(planes) if norm else nn.Identity(),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel,
                stride,
                upsample,
                activation,
                dropout,
                norm,
                residual,
                attn_on=attn_on,
                attn_depth=attn_depth,
                attn_heads=attn_heads,
            )
        )
        self.inplanes = planes # * 2 # CCCCC

        return nn.Sequential(*layers)

    def forward(self, x, residuals):
        for i in range(0, self.depth):
            if self.sym_residual:  # symmetric skip connection
                res = residuals[-1 - i]
                if res.ndim > x.ndim:  # for 3D to 2D
                    res = torch.mean(res, dim=2)

                # Element-wise addition of residual
                x = x + res * self.residual_scales[i]

                # Concatenation and fusion of residual
                # x = torch.concat((x, res), dim=1)
                # x = self.fusion_layers[str(i)](x)

            x = self.layers[str(i)](x)
        return x


class UNet1d(nn.Module):
    def __init__(
        self,
        depth: int = 6,
        channels_in: int = 1,
        channels: list = [1, 64, 128, 256, 256, 256, 256],
        kernels: list = [5, 3, 3, 3, 3, 3, 3],
        downsample: int = 4,
        attn: list = [0, 0, 0, 0, 0, 0, 0, 0, 0],
        attn_heads: int = 1,
        attn_depth: int = 1,
        dropout: float = 0.0,
        activation: Optional[Type[nn.Module]] = nn.ReLU,
        norm=True,
        fwd_skip: bool = True,
        sym_skip: bool = True,
    ) -> None:
        super().__init__()

        encoder_channels = [channels_in] + [channels] * depth if isinstance(channels, int) else channels
        encoder_channels = encoder_channels[0: depth + 1]
        decoder_channels = list(reversed(encoder_channels))
        decoder_channels[-1] = 1

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        # Automatically calculate the strides for each layer
        strides = [
            2 if i < int(np.log2(downsample)) else 1 for i in range(depth)
        ]

        self.encoder = AttnResNet1d(
            block=AttnResBlock1d,
            depth=depth,
            channels=self.encoder_channels,
            kernels=kernels,
            strides=strides,
            attn_on=attn,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=fwd_skip,
        )

        self.decoder = AttnResNet1dT(
            block=AttnResBlock1dT,
            depth=depth,
            channels=self.decoder_channels,
            kernels=list(reversed(kernels)),
            strides=list(reversed(strides)),
            # attn_on=list(reversed(attn[0:depth])),
            attn_on=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            sym_residual=sym_skip,
            fwd_residual=fwd_skip,
        )

    def forward(self, X: torch.Tensor):
        X = X.unsqueeze(1) if X.ndim < 4 else X
        Z, res = self.encoder(X)
        D = self.decoder(Z, res).squeeze(1)
        del X, Z, res
        return D