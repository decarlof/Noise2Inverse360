#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D U-Net for Noise2Inverse volumetric denoising.

Architecture source: SSD_3D (Laugros et al., bioRxiv 2025)
  "Self-supervised image restoration in coherent X-ray neuronal microscopy"
  https://doi.org/10.1101/2025.02.10.633538

Original U-Net implementation: ELEKTRONN3 (Martin Drawitsch, MPG)
  https://github.com/ELEKTRONN/elektronn3
  Based on https://github.com/jaxony/unet-pytorch (Jackson Huang, MIT License)

Modifications in this file:
  - Removed test utilities
  - Added unet3d() factory function for N2I-compatible instantiation
"""

__all__ = ['UNet', 'unet3d']

import copy

from typing import Sequence, Union, Tuple, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conv(dim=3):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_convtranspose(dim=3):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_maxpool(dim=3):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_normalization(normtype: str, num_channels: int, dim: int = 3):
    if normtype is None or normtype == 'none':
        return nn.Identity()
    elif normtype.startswith('group'):
        if normtype == 'group':
            num_groups = 8
        elif len(normtype) > len('group') and normtype[len('group'):].isdigit():
            num_groups = int(normtype[len('group'):])
        else:
            raise ValueError(
                f'normtype "{normtype}" not understood. Use "group<G>".'
            )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == 'instance':
        return nn.InstanceNorm3d(num_channels) if dim == 3 else nn.InstanceNorm2d(num_channels)
    elif normtype == 'batch':
        return nn.BatchNorm3d(num_channels) if dim == 3 else nn.BatchNorm2d(num_channels)
    elif normtype == 'layer':
        return nn.GroupNorm(1, num_channels=num_channels)
    else:
        raise ValueError(f'Unknown normalization type "{normtype}".')


def planar_kernel(x):
    return (1, x, x) if isinstance(x, int) else x


def planar_pad(x):
    return (0, x, x) if isinstance(x, int) else x


def conv3(in_channels, out_channels, kernel_size=3, stride=1,
          padding=1, bias=True, planar=False, dim=3):
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
    )


def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3):
    kernel_size = 2
    stride = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
    if mode == 'transpose':
        return get_convtranspose(dim)(in_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride)
    elif 'resizeconv' in mode:
        upsampling_mode = ('trilinear' if dim == 3 else 'bilinear') if 'linear' in mode else 'nearest'
        rc_kernel_size = 1 if mode.endswith('1') else 3
        return ResizeConv(in_channels, out_channels, planar=planar, dim=dim,
                          upsampling_mode=upsampling_mode, kernel_size=rc_kernel_size)


def conv1(in_channels, out_channels, dim=3):
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'lin':
            return nn.Identity()
    else:
        return copy.deepcopy(activation)


# ---------------------------------------------------------------------------
# Network blocks
# ---------------------------------------------------------------------------

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, planar=False,
                 activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same'):
        super().__init__()
        self.pooling = pooling
        self.dim = dim
        padding = 1 if 'same' in conv_mode else 0

        self.conv1 = conv3(in_channels, out_channels, planar=planar, dim=dim, padding=padding)
        self.conv2 = conv3(out_channels, out_channels, planar=planar, dim=dim, padding=padding)

        if self.pooling:
            kernel_size = planar_kernel(2) if planar else 2
            self.pool = get_maxpool(dim)(kernel_size=kernel_size, ceil_mode=True)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = -123

        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        self.norm0 = get_normalization(normalization, out_channels, dim=dim) if full_norm else nn.Identity()
        self.norm1 = get_normalization(normalization, out_channels, dim=dim)

    def forward(self, x):
        y = self.act1(self.norm0(self.conv1(x)))
        y = self.act2(self.norm1(self.conv2(y)))
        before_pool = y
        y = self.pool(y)
        return y, before_pool


@torch.jit.script
def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if from_down.shape[2:] == from_up.shape[2:]:
        return from_down, from_up

    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    upcrop = [u - ((u - d) % 2) for d, u in zip(ds, us)]

    ndim = from_down.dim()
    if ndim == 4:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1]]
    if ndim == 5:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1], :upcrop[2]]

    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    assert ds[0] >= us[0], f'{ds, us}'
    assert ds[1] >= us[1]
    if ndim == 4:
        from_down = from_down[
            :, :,
            (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2,
            (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2,
        ]
    elif ndim == 5:
        assert ds[2] >= us[2]
        from_down = from_down[
            :, :,
            (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2,
            (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2,
            (ds[2] - us[2]) // 2:(ds[2] + us[2]) // 2,
        ]
    return from_down, from_up


class DummyAttention(nn.Module):
    def forward(self, x, g):
        return x, None


class GridAttention(nn.Module):
    """Grid attention gate (Oktay et al., 2018)."""
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 dim=3, sub_sample_factor=2):
        super().__init__()
        assert dim in [2, 3]
        self.dim = dim
        self.sub_sample_factor = (sub_sample_factor,) * dim
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels or max(1, in_channels // 2)

        conv_nd = nn.Conv3d if dim == 3 else nn.Conv2d
        bn = nn.BatchNorm3d if dim == 3 else nn.BatchNorm2d
        self.upsample_mode = 'trilinear' if dim == 3 else 'bilinear'

        self.w = nn.Sequential(
            conv_nd(self.in_channels, self.in_channels, kernel_size=1),
            bn(self.in_channels),
        )
        self.theta = conv_nd(self.in_channels, self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size,
                             stride=self.sub_sample_factor, bias=False)
        self.phi = conv_nd(self.gating_channels, self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(self.inter_channels, 1, kernel_size=1, stride=1, bias=True)
        self.init_weights()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:],
                              mode=self.upsample_mode, align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.shape[2:],
                                   mode=self.upsample_mode, align_corners=False)
        return self.w(sigm_psi_f.expand_as(x) * x), sigm_psi_f

    def init_weights(self):
        def weight_init(m):
            cls = m.__class__.__name__
            if 'Conv' in cls:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif 'Linear' in cls:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif 'BatchNorm' in cls:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(weight_init)


class UpConv(nn.Module):
    att: Optional[torch.Tensor]

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose',
                 planar=False, activation='relu', normalization=None, full_norm=True,
                 dim=3, conv_mode='same', attention=False):
        super().__init__()
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        padding = 1 if 'same' in conv_mode else 0

        self.upconv = upconv2(in_channels, out_channels, mode=up_mode, planar=planar, dim=dim)

        in1 = 2 * out_channels if merge_mode == 'concat' else out_channels
        self.conv1 = conv3(in1, out_channels, planar=planar, dim=dim, padding=padding)
        self.conv2 = conv3(out_channels, out_channels, planar=planar, dim=dim, padding=padding)

        self.act0 = get_activation(activation)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, out_channels, dim=dim)
            self.norm1 = get_normalization(normalization, out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        self.norm2 = get_normalization(normalization, out_channels, dim=dim)

        self.attention = GridAttention(in_channels // 2, in_channels, dim=dim) if attention else DummyAttention()
        self.att = None

    def forward(self, enc, dec):
        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        if not torch.jit.is_scripting():
            self.att = att
        updec = self.act0(self.norm0(updec))
        mrg = torch.cat((updec, genc), 1) if self.merge_mode == 'concat' else updec + genc
        y = self.act1(self.norm1(self.conv1(mrg)))
        y = self.act2(self.norm2(self.conv2(y)))
        return y


class ResizeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False,
                 dim=3, upsampling_mode='nearest'):
        super().__init__()
        self.scale_factor = planar_kernel(2) if (dim == 3 and planar) else 2
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=upsampling_mode)
        if kernel_size == 3:
            self.conv = conv3(in_channels, out_channels, padding=1, planar=planar, dim=dim)
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(f'kernel_size={kernel_size} not supported.')

    def forward(self, x):
        return self.conv(self.upsample(x))


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    3D U-Net with skip connections for volumetric image restoration.

    Input:  [B, in_channels,  D, H, W]
    Output: [B, out_channels, D, H, W]

    Spatial dimensions must each be divisible by 2**n_blocks.
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            n_blocks: int = 3,
            start_filts: int = 32,
            up_mode: str = 'transpose',
            merge_mode: str = 'concat',
            planar_blocks: Sequence = (),
            batch_norm: str = 'unset',
            attention: bool = False,
            activation: Union[str, nn.Module] = 'relu',
            normalization: str = 'layer',
            full_norm: bool = True,
            dim: int = 3,
            conv_mode: str = 'same',
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError('n_blocks must be >= 1.')
        if dim not in {2, 3}:
            raise ValueError('dim must be 2 or 3.')
        if batch_norm != 'unset':
            raise RuntimeError('Use normalization= instead of batch_norm=.')
        if up_mode not in ('transpose', 'resizeconv_nearest', 'resizeconv_linear',
                           'resizeconv_nearest1', 'resizeconv_linear1'):
            raise ValueError(f'Invalid up_mode: {up_mode}')
        if merge_mode not in ('concat', 'add'):
            raise ValueError(f'Invalid merge_mode: {merge_mode}')

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.planar_blocks = planar_blocks
        self.normalization = normalization
        self.attention = attention
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim
        self.up_mode = up_mode
        self.merge_mode = merge_mode

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Encoder
        outs = None
        for i in range(n_blocks):
            ins = in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            planar = i in planar_blocks
            self.down_convs.append(DownConv(
                ins, outs,
                pooling=(i < n_blocks - 1),
                planar=planar,
                activation=activation,
                normalization=normalization,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            ))

        # Decoder
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = (n_blocks - 2 - i) in planar_blocks
            self.up_convs.append(UpConv(
                ins, outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                planar=planar,
                activation=activation,
                normalization=normalization,
                attention=attention,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            ))

        self.conv_final = conv1(outs, out_channels, dim=dim)
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, GridAttention):
            return
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            x = module(encoder_outs[-(i + 2)], x)
        return self.conv_final(x)

    @torch.jit.unused
    def forward_gradcp(self, x):
        """Forward pass with gradient checkpointing (saves ~20-50% memory)."""
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = checkpoint(module, x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            x = checkpoint(module, encoder_outs[-(i + 2)], x)
        return self.conv_final(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def unet3d(
    in_channels: int = 1,
    out_channels: int = 1,
    n_blocks: int = 3,
    start_filts: int = 32,
    normalization: str = 'layer',
    activation: str = 'relu',
) -> UNet:
    """
    Factory function matching the configuration used in Laugros et al. 2025.

    Input/output shape: [B, 1, D, H, W]
    D, H, W must each be divisible by 2**n_blocks (e.g. 64 for n_blocks=3).

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for single-channel volumes).
    out_channels : int
        Number of output channels (1 for denoising).
    n_blocks : int
        Encoder depth. Controls receptive field and GPU memory use.
        Default 3 is a good balance; use 4 for deeper context (needs more memory).
    start_filts : int
        Filter count of the first encoder block. Subsequent blocks double this.
    normalization : str
        Normalisation type: 'layer', 'group', 'instance', 'batch', or 'none'.
    activation : str
        Activation function: 'relu', 'leaky', 'silu', etc.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        start_filts=start_filts,
        up_mode='transpose',
        merge_mode='concat',
        planar_blocks=(),
        attention=False,
        activation=activation,
        normalization=normalization,
        full_norm=True,
        dim=3,
        conv_mode='same',
    )
