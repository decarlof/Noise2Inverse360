#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D dataset classes for Noise2Inverse volumetric denoising (--mode 3d).

Training  : TomoDataset3DTrain  — random cubic patch sampling with full 3D
                                   geometric augmentation (24 rotational
                                   symmetries of a cube + random flip).
Inference : TomoDataset3DInfer  — sliding-window 3D grid extraction with
                                   overlap-add stitching back to full volume.

Augmentation strategy adapted from SSD_3D (Laugros et al., bioRxiv 2025).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Literal, Optional, Tuple

from denoise import log
from denoise import tiffs


# ---------------------------------------------------------------------------
# Normalization helper (mirrors data.py)
# ---------------------------------------------------------------------------

def save_normalization_value_3d(config_file, mean, std):
    """Write mean4norm / std4norm into the YAML config (reuses data.py's version)."""
    from denoise.data import save_normalization_value
    save_normalization_value(config_file, mean, std)


# ---------------------------------------------------------------------------
# 3D geometric augmentation (24 rotational symmetries of a cube + h-flip)
# Ported from SSD_3D/datasets.py (Laugros et al., 2025)
# ---------------------------------------------------------------------------

def geom_transform_3d(vol0: torch.Tensor, vol1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply one of the 24 rotational symmetries of a cube, then optionally
    flip horizontally.  The *same* transform is applied to both volumes so
    the N2I pairing is preserved.

    Parameters
    ----------
    vol0, vol1 : torch.Tensor, shape [1, D, H, W]

    Returns
    -------
    vol0, vol1 : torch.Tensor, shape [1, D, H, W]  (transformed)
    """
    direction      = torch.randint(0, 3, (1,)).item()
    channel_flip   = torch.randint(0, 2, (1,)).item()
    nb_rot         = torch.randint(0, 4, (1,)).item()
    h_flip         = torch.randint(0, 2, (1,)).item()

    def _apply(v):
        v = v.squeeze(0)                     # [D, H, W]
        if direction == 1:
            v = v.permute(1, 0, 2)          # swap D <-> H
        elif direction == 2:
            v = v.permute(2, 1, 0)          # swap D <-> W
        if channel_flip:
            v = torch.flip(v, [0])          # flip along depth
        v = torch.rot90(v, nb_rot, [1, 2])  # rotate in H-W plane
        if h_flip:
            v = torch.flip(v, [2])          # flip W
        return v.unsqueeze(0)               # [1, D, H, W]

    return _apply(vol0), _apply(vol1)


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class TomoDataset3DTrain(Dataset):
    """
    Training dataset for 3D Noise2Inverse.

    Loads two TIFF stacks (split0 / split1) into CPU memory as 3D NumPy
    arrays [D, H, W], normalises them, and serves random cubic patches of
    size psz_3d with 3D geometric augmentation.

    Parameters
    ----------
    params : dict
        Parsed YAML config (same structure as 2.5D).
    config_file : str
        Path to the YAML, used to save normalisation stats.
    """

    def __init__(self, params: dict, config_file: str):
        super().__init__()
        dp = params['dataset']
        tp = params['train']

        self.psz        = int(tp.get('psz_3d', tp.get('psz', 64)))
        self.n_patches  = int(tp.get('nb_patches_3d', 1000))

        recon_0 = dp['directory_to_reconstructions'] + '/' + dp['sub_recon_name0']
        recon_1 = dp['directory_to_reconstructions'] + '/' + dp['sub_recon_name1']

        log.info("3D train: loading split0 from %s" % recon_0)
        tiffs0 = tiffs.glob(recon_0)
        self.split0, mean0, std0 = tiffs.load_stack(tiffs0)

        log.info("3D train: loading split1 from %s" % recon_1)
        tiffs1 = tiffs.glob(recon_1)
        self.split1, mean1, std1 = tiffs.load_stack(tiffs1)

        self.split0 = ((self.split0 - mean0) / std0).astype(np.float32)
        self.split1 = ((self.split1 - mean1) / std1).astype(np.float32)

        log.info("3D split0  mean=%.4f  std=%.4f  shape=%s" % (mean0, std0, self.split0.shape))
        log.info("3D split1  mean=%.4f  std=%.4f  shape=%s" % (mean1, std1, self.split1.shape))

        # Expose stats so train.py can save them
        self.split0_mean = float(mean0)
        self.split0_std  = float(std0)

        D, H, W = self.split0.shape
        if self.psz > min(D, H, W):
            raise ValueError(
                "psz_3d=%d exceeds volume dimension (D=%d H=%d W=%d). "
                "Reduce psz_3d in the YAML config." % (self.psz, D, H, W)
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, _idx):
        D, H, W = self.split0.shape
        psz = self.psz

        # Random cubic patch origin
        d0 = np.random.randint(0, D - psz + 1)
        h0 = np.random.randint(0, H - psz + 1)
        w0 = np.random.randint(0, W - psz + 1)

        patch0 = self.split0[d0:d0+psz, h0:h0+psz, w0:w0+psz].copy()
        patch1 = self.split1[d0:d0+psz, h0:h0+psz, w0:w0+psz].copy()

        # [1, D, H, W]
        t0 = torch.from_numpy(patch0).unsqueeze(0)
        t1 = torch.from_numpy(patch1).unsqueeze(0)

        # Apply identical 3D geometric augmentation to both patches
        t0, t1 = geom_transform_3d(t0, t1)

        return t0, t1   # each [1, psz, psz, psz]


# ---------------------------------------------------------------------------
# Inference dataset
# ---------------------------------------------------------------------------

def _positions_3d(length: int, patch: int, stride: int) -> List[int]:
    """Sliding-window positions that always cover the last partial patch."""
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _hann3d(pd: int, ph: int, pw: int, eps: float = 1e-6) -> np.ndarray:
    """3D Hann blending window [pd, ph, pw]."""
    wd = np.hanning(pd).astype(np.float32)
    wh = np.hanning(ph).astype(np.float32)
    ww = np.hanning(pw).astype(np.float32)
    wd = np.maximum(wd, eps)
    wh = np.maximum(wh, eps)
    ww = np.maximum(ww, eps)
    return wd[:, None, None] * wh[None, :, None] * ww[None, None, :]


class TomoDataset3DInfer(Dataset):
    """
    Inference dataset for 3D Noise2Inverse.

    Loads a TIFF stack [D, H, W] and yields overlapping cubic patches for
    model inference.  After all patches are predicted, call
    ``stitch_predictions()`` to blend them back into the full volume.

    Parameters
    ----------
    params : dict
        Parsed YAML config.
    start_slice : str
        First slice index (empty string = first slice).
    end_slice : str or None
        Last slice index (None = last slice).
    """

    def __init__(
        self,
        params: dict,
        start_slice: str = '',
        end_slice: Optional[str] = None,
    ):
        super().__init__()
        dp = params['dataset']
        tp = params['train']
        ip = params['infer']

        recon_dir = dp['directory_to_reconstructions'] + '/' + dp['full_recon_name']
        self.psz     = int(tp.get('psz_3d', tp.get('psz', 64)))
        overlap      = float(ip.get('overlap', 0.5))
        mean4norm    = float(dp['mean4norm'])
        std4norm     = float(dp['std4norm'])

        tiffs_col = tiffs.glob(recon_dir)
        if start_slice:
            tiffs_col = tiffs_col[int(start_slice):int(end_slice)]

        log.info("3D infer: loading volume from %s" % recon_dir)
        vol, _, _ = tiffs.load_stack(tiffs_col)
        self.vol = ((vol - mean4norm) / std4norm).astype(np.float32)
        log.info("3D infer: volume shape %s" % str(self.vol.shape))

        D, H, W = self.vol.shape
        self.D, self.H, self.W = D, H, W
        self.start_slice = int(start_slice) if start_slice else 0

        stride = max(1, int(round(self.psz * (1.0 - overlap))))
        self.stride = stride
        self.overlap = overlap

        # Pad volume so patches fit without going out of bounds
        def _pad(n, p, s):
            if n >= p:
                extra = (p - (n - p) % s) % s if (n - p) % s != 0 else 0
                return n + extra
            return p

        self.D_pad = _pad(D, self.psz, stride)
        self.H_pad = _pad(H, self.psz, stride)
        self.W_pad = _pad(W, self.psz, stride)

        # Pad the volume once
        pad_d = self.D_pad - D
        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        self.vol_padded = np.pad(
            self.vol,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode='reflect'
        )

        self.d_pos = _positions_3d(self.D_pad, self.psz, stride)
        self.h_pos = _positions_3d(self.H_pad, self.psz, stride)
        self.w_pos = _positions_3d(self.W_pad, self.psz, stride)

        # Flat index of all patches
        self.index: List[Tuple[int, int, int]] = [
            (d, h, w)
            for d in self.d_pos
            for h in self.h_pos
            for w in self.w_pos
        ]

        log.info("3D infer: %d patches  (stride=%d, overlap=%.2f)" % (
            len(self.index), stride, overlap))

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        d, h, w = self.index[i]
        psz = self.psz
        patch = self.vol_padded[d:d+psz, h:h+psz, w:w+psz].copy()
        return torch.from_numpy(patch).unsqueeze(0)   # [1, psz, psz, psz]

    # ------------------------------------------------------------------

    def stitch_predictions(
        self,
        pred_patches: np.ndarray,
        window: Literal['uniform', 'hann'] = 'hann',
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Overlap-add stitching of 3D patch predictions.

        Parameters
        ----------
        pred_patches : np.ndarray, shape [T, psz, psz, psz]
            Model outputs in the same order as dataset iteration.
        window : str
            Blending window: 'hann' (recommended) or 'uniform'.
        eps : float
            Denominator safety term.

        Returns
        -------
        vol : np.ndarray, shape [D, H, W]  (original unpadded size)
        """
        if pred_patches.shape[0] != len(self.index):
            raise ValueError(
                "Expected %d patches, got %d." % (len(self.index), pred_patches.shape[0])
            )

        psz = self.psz
        acc  = np.zeros((self.D_pad, self.H_pad, self.W_pad), dtype=np.float32)
        wacc = np.zeros_like(acc)

        if window == 'hann':
            w3d = _hann3d(psz, psz, psz, eps=eps)
        else:
            w3d = np.ones((psz, psz, psz), dtype=np.float32)

        for t, (d, h, w) in enumerate(self.index):
            patch = pred_patches[t]           # [psz, psz, psz]
            acc [d:d+psz, h:h+psz, w:w+psz] += patch * w3d
            wacc[d:d+psz, h:h+psz, w:w+psz] += w3d

        out = acc / (wacc + eps)
        return out[:self.D, :self.H, :self.W].copy()
