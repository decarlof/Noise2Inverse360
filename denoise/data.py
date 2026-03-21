import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import yaml, sys
import warnings
import torch
import torch.nn as nn

from denoise import log
from denoise import tiffs

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

def save_normalization_value(config_file, mean, std):
        """
        This functin saves the mean and standard deviation back to the yaml file which is then used during inferencing

        params:
            - config_file (str) location of the config file
            - mean (float) mean used for normalization
            - std (float) standard deviation used for normalization
        """
        # safe load
        try:
            with open(config_file, 'r') as file:
                data = yaml.safe_load(file) # Use safe_load for security
        except FileNotFoundError:
            data = {} # If the file doesn't exist, start with an empty dictionary
        except yaml.YAMLError as exc:
            log.error("Error loading YAML file: %s" % exc)
            data = {} # Handle parsing errors

        data['dataset']['mean4norm'] = float(mean)
        data['dataset']['std4norm'] = float(std)

        #write the data back to the yaml file
        with open(config_file, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)

class TomoDatasetTrain(Dataset):
    """
    Training class for 2.5D N2I.

    This class loads in two lists corresponding to the two sub reconstructions
    (saved as .tiffs) and normalizes them.

    params:
        - params (obj) yaml object, essentially a dictionary
        - config_file (str) location of the configuration file
    """
    def __init__(self, params, config_file):
        super(TomoDatasetTrain, self).__init__()
        dataset_params = params['dataset']
        train_params = params['train']

        # input image patch size
        self.psz = train_params['psz']

        # number of adjacent slices to use
        self.n_slices = train_params['n_slices']
        z_stride = train_params.get('z_stride', 1)
        
        # specify augmentations for training
        self.augmentations = A.Compose([
                A.SquareSymmetry(p=1.0),
        ],
        additional_targets={'split1': 'image'}
        )

        # load in tiff images for training

        # location to sub reconstructions
        recon_0_path =  dataset_params['directory_to_reconstructions'] + '/' + dataset_params['sub_recon_name0']
        recon_1_path =  dataset_params['directory_to_reconstructions'] + '/' + dataset_params['sub_recon_name1']

        # collect tiff files and load in images
        tiffs_collection = tiffs.glob(recon_0_path)[::z_stride]
        self.split0, split0_mean, split0_std = tiffs.load_stack(tiffs_collection)

        tiffs_collection = tiffs.glob(recon_1_path)[::z_stride]
        self.split1, split1_mean, split1_std = tiffs.load_stack(tiffs_collection)

        # normalize the data
        self.split0 -= split0_mean
        self.split0 /= split0_std
        log.info(f"\nSplit 0 is scaled with calculated mean: {split0_mean}, std: {split0_std}")

        self.split1 -= split1_mean
        self.split1 /= split1_std
        log.info(f"\nSplit 1 is scaled with calculated mean: {split1_mean}, std: {split1_std}")
    
        self.split0_mean = split0_mean
        self.split0_std  = split0_std

        self.samples = self.__len__()
    
    def __getitem__(self, idx):


        S, H, W = int(self.split0.shape[0]), int(self.split0.shape[1]), int(self.split0.shape[2])

        # compute offsets around the center
        left = self.n_slices // 2
        right = self.n_slices - 1 - left
        offsets = np.arange(-left, right + 1, dtype=int)  # length == num_slices
        idxs = idx + offsets
        idxs_mapped = np.clip(idxs, 0, S - 1)

        view0 = self.split0[idxs_mapped] 
        view1 = self.split1[idxs_mapped] 

        #randomly select patch size
        rst = np.random.randint(0, self.split0.shape[-2]-self.psz)
        cst = np.random.randint(0, self.split0.shape[-1]-self.psz)

        view0 = view0[:, rst:rst+self.psz, cst:cst+self.psz]
        view1 = view1[:, rst:rst+self.psz, cst:cst+self.psz]

        # perform augmentations
        augmented = self.augmentations(image=np.moveaxis(view0, 0, -1), split1=np.moveaxis(view1, 0, -1))
        view0 = np.moveaxis(augmented['image'], -1, 0)
        view1 = np.moveaxis(augmented['split1'], -1, 0)

        return view0, view1

    def __len__(self):
        return self.split0.shape[0]
    
def _compute_positions(length: int, patch: int, stride: int) -> List[int]:
    """
    Sliding-window positions that always include the last patch touching the boundary.
    """
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _pad_hw_numpy(
    arr: np.ndarray,
    pad_bottom: int,
    pad_right: int,
    mode: Literal["reflect", "edge", "constant"] = "reflect",
    constant_values: float = 0.0,
) -> np.ndarray:
    """
    Pad a 2D or 3D array on H/W (last two dims) using numpy.pad.
    - arr can be [H,W] or [C,H,W]
    """
    if pad_bottom == 0 and pad_right == 0:
        return arr

    if arr.ndim == 2:
        pad_width = ((0, pad_bottom), (0, pad_right))
    elif arr.ndim == 3:
        pad_width = ((0, 0), (0, pad_bottom), (0, pad_right))
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    if mode == "constant":
        return np.pad(arr, pad_width, mode=mode, constant_values=constant_values)
    return np.pad(arr, pad_width, mode=mode)


def _build_2p5d_stack(
    vol: np.ndarray,
    d_idx: int,
    neighbors: int,
    edge_mode: Literal["reflect", "edge", "constant"] = "reflect",
    constant_values: float = 0.0,
) -> np.ndarray:
    """
    Build a 2.5D stack centered at slice d_idx:
      returns stack shape [C, H, W], where C = 2*neighbors + 1.

    Edge handling:
      - "reflect": reflect indices at boundaries
      - "edge": clamp to [0, D-1]
      - "constant": out-of-bounds slices filled with constant_values
    """
    assert vol.ndim == 3, f"Expected vol [D,H,W], got {vol.shape}"
    D, H, W = vol.shape
    C = 2 * neighbors + 1

    if edge_mode == "constant":
        stack = np.full((C, H, W), constant_values, dtype=vol.dtype)
        for ci, off in enumerate(range(-neighbors, neighbors + 1)):
            di = d_idx + off
            if 0 <= di < D:
                stack[ci] = vol[di]
        return stack

    def reflect_index(i: int, n: int) -> int:
        # Reflect around boundaries for i outside [0, n-1]
        # Example for n=5: valid 0..4
        # -1 -> 1, -2 -> 2, 5 -> 3, 6 -> 2, ...
        if n == 1:
            return 0
        while i < 0 or i >= n:
            if i < 0:
                i = -i
            if i >= n:
                i = 2 * (n - 1) - i
        return i

    stack = np.empty((C, H, W), dtype=vol.dtype)
    for ci, off in enumerate(range(-neighbors, neighbors + 1)):
        di = d_idx + off
        if edge_mode == "edge":
            di = min(max(di, 0), D - 1)
        elif edge_mode == "reflect":
            di = reflect_index(di, D)
        else:
            raise ValueError("edge_mode must be one of: 'reflect', 'edge', 'constant'")
        stack[ci] = vol[di]
    return stack


def _make_blend_window_np(
    ph: int,
    pw: int,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Returns a 2D blending window [ph, pw] for overlap-add stitching.
    """
    if window == "uniform":
        w2d = np.ones((ph, pw), dtype=np.float32)
    elif window in ("hann", "cosine"):
        wh = np.hanning(ph).astype(np.float32)
        ww = np.hanning(pw).astype(np.float32)
        # avoid exact zeros at the ends (helps if a pixel only lands on borders due to edge cases)
        wh = np.maximum(wh, eps)
        ww = np.maximum(ww, eps)
        w2d = wh[:, None] * ww[None, :]
    else:
        raise ValueError("window must be one of: 'uniform', 'hann', 'cosine'")
    return w2d

@dataclass(frozen=True)
class PatchIndex:
    """
    One patch address within the volume.
    d_idx: center slice index for 2.5D stack
    top,left: spatial location within padded H/W coordinates
    """
    d_idx: int
    top: int
    left: int


@dataclass
class TilingMeta:
    """
    Metadata needed for stitching (later) and sanity checks.
    """
    D: int
    H_in: int
    W_in: int
    H_pad: int
    W_pad: int
    ph: int
    pw: int
    stride_h: int
    stride_w: int
    n_rows: int
    n_cols: int
    P_per_slice: int
    pad_bottom: int
    pad_right: int
    neighbors: int
    edge_mode: str
    pad_mode: str


class TomoDatasetInfer(Dataset):
    """
    Dataset that yields overlapping patches from a CT volume (NumPy) with 2.5D channels.

    Input volume: vol [D, H, W] (NumPy array)

    Each item returns:

    - x_patch: torch.FloatTensor [C, ph, pw]
    - info: dict with patch coordinates and indices (for stitching later)
    """
    def __init__(
        self,
        params: dict,
        start_slice: int = 0,
        end_slice: int = None,
        slice_range: Optional[Tuple[int, int]] = None,
        output_dtype: np.dtype = np.float32,
        pad_mode: Literal["reflect", "edge", "constant"] = "reflect",
        pad_constant: float = 0.0,
        edge_mode: Literal["reflect", "edge", "constant"] = "reflect",
        edge_constant: float = 0.0,
        return_info: bool = True,
    ):
        """
        params:

            params (obj) yaml object, essentially a dictionary
            start_slice: start slice of volume
            end_slice: end slice of volume
            slice_range: (d_start, d_end) inclusive/exclusive, like Python slicing.
                         If None, uses full [0, D).
            preprocess: optional function applied to vol once at init (e.g., windowing/scaling).
                        Signature: vol_np -> vol_np (still [D,H,W]).
            output_dtype: dtype for patches (np.float32 recommended).
            pad_mode: padding for H/W when patch doesn't fit exactly:
                      "reflect", "edge" (replicate), or "constant".
            pad_constant: used if pad_mode="constant".
            edge_mode: how to handle 2.5D neighbors near D boundaries:
                       "reflect", "edge", or "constant".
            edge_constant: used if edge_mode="constant".
            return_info: if True, returns dict with (d_idx, top, left) and meta.
        """
        super().__init__()

        dataset_params = params['dataset']
        recon_dir = dataset_params['directory_to_reconstructions'] + '/' + dataset_params['full_recon_name']
        patch_size = (params['train']['psz'], params['train']['psz'])
        overlap = params['infer']['overlap']
        neighbors = int(params['train']['n_slices'] // 2)
        mean4norm = params['dataset']['mean4norm']
        std4norm = params['dataset']['std4norm']

        # process slice if specified
        if len(start_slice) == 0:
            tiffs_collection = tiffs.glob(recon_dir)
        else:
            tiffs_collection = tiffs.glob(recon_dir)[int(start_slice):int(end_slice)]
        
        #print(tiffs_collection)
        self.vol, _, _ = tiffs.load_stack(tiffs_collection)

        self.vol -= mean4norm
        self.vol /= std4norm
        log.info(f'Volume Size: {self.vol.shape}')
        log.info(f"\nReconstruction is scaled with provided mean: {mean4norm}, std: {std4norm}")

        self.vol = self.vol.astype(output_dtype, copy=False)

        self.ph, self.pw = patch_size
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1).")
        self.overlap = float(overlap)
        self.neighbors = int(neighbors)
        self.return_info = bool(return_info)

        D, H, W = self.vol.shape
        self.D, self.H_in, self.W_in = D, H, W

        # Slice range
        if slice_range is None:
            d_start, d_end = 0, D
        else:
            d_start, d_end = slice_range
            d_start = max(0, int(d_start))
            d_end = min(D, int(d_end))
            if d_end <= d_start:
                raise ValueError("slice_range must satisfy d_end > d_start")
        self.d_start, self.d_end = d_start, d_end

        # Strides
        self.stride_h = max(1, int(round(self.ph * (1.0 - self.overlap))))
        self.stride_w = max(1, int(round(self.pw * (1.0 - self.overlap))))

        # Pad bottom/right so patches fit at least once
        H_needed = max(H, self.ph)
        W_needed = max(W, self.pw)
        self.pad_bottom = H_needed - H
        self.pad_right = W_needed - W

        # Create padded view of each 2D slice on-the-fly (we’ll pad the stack per item),
        # but we also compute positions on the padded shape:
        H_pad = H + self.pad_bottom
        W_pad = W + self.pad_right
        self.H_pad, self.W_pad = H_pad, W_pad

        # Pre-pad H/W once so __getitem__ can index patches directly without
        # copying full slices (reduces per-item memory movement from ~200 MB to ~1 MB).
        self.vol_padded = _pad_hw_numpy(self.vol, self.pad_bottom, self.pad_right,
                                        mode=pad_mode, constant_values=pad_constant)

        self.top_positions = _compute_positions(H_pad, self.ph, self.stride_h)
        self.left_positions = _compute_positions(W_pad, self.pw, self.stride_w)

        self.n_rows = len(self.top_positions)
        self.n_cols = len(self.left_positions)
        self.P_per_slice = self.n_rows * self.n_cols
        self.total_patches = (self.d_end - self.d_start) * self.P_per_slice

        self.pad_mode = pad_mode
        self.pad_constant = float(pad_constant)
        self.edge_mode = edge_mode
        self.edge_constant = float(edge_constant)

        # Build a flat index of all patches across requested slices
        self.index: List[PatchIndex] = []
        for d_idx in range(self.d_start, self.d_end):
            for top in self.top_positions:
                for left in self.left_positions:
                    self.index.append(PatchIndex(d_idx=d_idx, top=top, left=left))

        self.meta = TilingMeta(
            D=D,
            H_in=H,
            W_in=W,
            H_pad=H_pad,
            W_pad=W_pad,
            ph=self.ph,
            pw=self.pw,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            P_per_slice=self.P_per_slice,
            pad_bottom=self.pad_bottom,
            pad_right=self.pad_right,
            neighbors=self.neighbors,
            edge_mode=self.edge_mode,
            pad_mode=self.pad_mode,
        )


    def stitch_predictions(
        self,
        pred_patches: np.ndarray,
        *,
        window: Literal["uniform", "hann", "cosine"] = "hann",
        output_size: Optional[Tuple[int, int]] = None,
        keep_k_dim: bool = True,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Stitch patch predictions back into a (sub)volume using overlap-add blending.

        Args:
            pred_patches:
                Patch predictions in dataset order (same order as self.index):
                  - [T, ph, pw]               regression (implicit K=1)
                  - [T, 1, ph, pw]            regression (explicit K=1)
                  - [T, K, ph, pw]            segmentation logits/probs

                where T must equal len(self.index) == total_patches.

            window:
                Blending window: 'uniform' (avg) or 'hann'/'cosine' (recommended).

            output_size:
                (H, W) to crop final result. If None, uses (H_in, W_in).

            keep_k_dim:
                If pred_patches was [T, ph, pw], output can be:
                  - keep_k_dim=True  -> [D_sel, 1, H, W]
                  - keep_k_dim=False -> [D_sel, H, W]

                If pred_patches already has K dim, output keeps it.

            eps:
                Small constant to avoid divide-by-zero in normalization.

        Returns:
            out:
              - [D_sel, K, H, W] (typical)
              - or [D_sel, H, W] if K==1 and keep_k_dim=False

            where D_sel = (d_end - d_start), i.e., only the slices this dataset processed.
            These correspond to the *center slice predictions* in your 2.5D setup.
        """
        T_expected = len(self.index)
        if pred_patches.shape[0] != T_expected:
            raise ValueError(
                f"pred_patches has T={pred_patches.shape[0]} but dataset expects T={T_expected}. "
                f"Make sure predictions are stored in the same order as dataset iteration."
            )

        ph, pw = self.ph, self.pw
        H_pad, W_pad = self.H_pad, self.W_pad

        # Normalize shapes to [T, K, ph, pw]
        if pred_patches.ndim == 3:
            # [T, ph, pw] -> [T, 1, ph, pw]
            pred = pred_patches[:, None, :, :]
            K = 1
        elif pred_patches.ndim == 4:
            # [T, K, ph, pw]
            pred = pred_patches
            K = pred.shape[1]
        else:
            raise ValueError(f"pred_patches must have 3 or 4 dims, got shape {pred_patches.shape}")

        if pred.shape[-2:] != (ph, pw):
            raise ValueError(
                f"Patch spatial size mismatch: pred has {pred.shape[-2:]}, expected {(ph, pw)}"
            )

        # Determine output cropping
        if output_size is None:
            H_out, W_out = self.H_in, self.W_in
        else:
            H_out, W_out = output_size

        D_sel = self.d_end - self.d_start

        # Accumulators (float32 is usually fine; float64 if you want ultra-safe accumulation)
        acc = np.zeros((D_sel, K, H_pad, W_pad), dtype=np.float32)
        wacc = np.zeros((D_sel, 1, H_pad, W_pad), dtype=np.float32)

        w2d = _make_blend_window_np(ph, pw, window=window, eps=eps)  # [ph, pw]
        
        # Stitch in the same order as self.index
        for t, pi in enumerate(self.index):
            d_local = pi.d_idx - self.d_start  # map global slice idx -> [0..D_sel-1]
            top, left = pi.top, pi.left

            patch = pred[t]  # [K, ph, pw]
            acc[d_local, :, top:top + ph, left:left + pw] += patch * w2d[None, :, :]
            wacc[d_local, :, top:top + ph, left:left + pw] += w2d[None, :, :]

        out = acc / (wacc + eps)  # [D_sel, K, H_pad, W_pad]
        out = out[:, :, :H_out, :W_out].copy()

        # Optionally squeeze K dim for regression convenience
        if K == 1 and (pred_patches.ndim == 3) and not keep_k_dim:
            out = out[:, 0]  # [D_sel, H, W]

        return out

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        pi = self.index[i]
        D = self.D
        neighbors = self.neighbors
        top, left = pi.top, pi.left

        # Build patch [C, ph, pw] by indexing directly into the pre-padded volume.
        # This avoids copying full slices (~200 MB) just to extract a small patch.
        if self.edge_mode == 'constant':
            C = 2 * neighbors + 1
            patch = np.full((C, self.ph, self.pw), self.edge_constant, dtype=self.vol_padded.dtype)
            for ci, off in enumerate(range(-neighbors, neighbors + 1)):
                di = pi.d_idx + off
                if 0 <= di < D:
                    patch[ci] = self.vol_padded[di, top:top + self.ph, left:left + self.pw]
        else:
            di_list = []
            for off in range(-neighbors, neighbors + 1):
                di = pi.d_idx + off
                if self.edge_mode == 'edge':
                    di = min(max(di, 0), D - 1)
                else:  # reflect
                    while di < 0 or di >= D:
                        if di < 0:
                            di = -di
                        if di >= D:
                            di = 2 * (D - 1) - di
                di_list.append(di)
            patch = self.vol_padded[di_list, top:top + self.ph, left:left + self.pw]

        if not self.return_info:
            return patch

        info = {
            "d_idx": pi.d_idx,
            "top": top,
            "left": left,
            "ph": self.ph,
            "pw": self.pw,
            "neighbors": neighbors,
            "C": 2 * neighbors + 1,
            "H_in": self.H_in,
            "W_in": self.W_in,
            "H_pad": self.H_pad,
            "W_pad": self.W_pad,
            "pad_bottom": self.pad_bottom,
            "pad_right": self.pad_right,
            "stride_h": self.stride_h,
            "stride_w": self.stride_w,
            "P_per_slice": self.P_per_slice,
            "d_start": self.d_start,
            "d_end": self.d_end,
        }
        return patch, info