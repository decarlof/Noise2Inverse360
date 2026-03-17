import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional, Literal, Union

def extract_sliding_window_patches_25d(
    x: torch.Tensor,
    patch_size: Tuple[int, int] = (512, 512),
    overlap: float = 0.5,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
    return_coords: bool = True,
) -> Tuple[torch.Tensor, Optional[List[Tuple[int, int]]], Dict[str, int]]:
    """
    Extract sliding-window patches from a 2.5D CT input tensor.

    Input shape:
        x: [N, C, H, W]
          - N: number of samples/windows (e.g., number of center slices you are inferring)
          - C: "channels" = number of adjacent slices in your 2.5D stack
          - H, W: spatial size of each slice

    IMPORTANT (alignment guarantee):
        Patches are extracted with the SAME (top,left) coordinates across ALL channels C.
        So neighbors remain perfectly aligned within each [C, h, w] patch.

    Args:
        x: Tensor [N, C, H, W]
        patch_size: (ph, pw)
        overlap: fraction in [0, 1). overlap=0.5 => stride = 0.5 * patch_size.
                 You may also pass overlap=0.0 for non-overlapping.
        pad_mode: padding mode for edges. Options include:
                  "reflect", "replicate", "constant".
        pad_value: used only if pad_mode="constant".
        return_coords: if True, return list of (top,left) coords for each patch.

    Returns:
        patches: Tensor [N, P, C, ph, pw]
            where P is number of spatial patches per image.

        coords: List[(top,left)] of length P (shared across N). None if return_coords=False.

        meta: dict with useful info:

            - H_in, W_in: original spatial size
            - H_pad, W_pad: padded spatial size
            - ph, pw: patch size
            - stride_h, stride_w
            - n_rows, n_cols
            - P: number of patches per image
            - pad_top, pad_left (always 0 here; we pad bottom/right for simplicity)
            - pad_bottom, pad_right
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [N,C,H,W], got {tuple(x.shape)}")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    N, C, H, W = x.shape
    ph, pw = patch_size
    if ph <= 0 or pw <= 0:
        raise ValueError("patch_size must be positive.")

    # Stride derived from overlap
    stride_h = max(1, int(round(ph * (1.0 - overlap))))
    stride_w = max(1, int(round(pw * (1.0 - overlap))))

    # Ensure we cover the full image with a last patch that reaches the boundary.
    # We'll compute the grid of top/left coordinates on the *padded* tensor.
    # To avoid negative behavior, we pad bottom/right so that (last_top + ph == H_pad) and similarly for W.
    def _compute_positions(length: int, patch: int, stride: int) -> List[int]:
        if length <= patch:
            return [0]
        positions = list(range(0, length - patch + 1, stride))
        if positions[-1] != length - patch:
            positions.append(length - patch)
        return positions

    # First compute positions on original size; then pad if needed so that patch fits.
    top_positions = _compute_positions(H, ph, stride_h)
    left_positions = _compute_positions(W, pw, stride_w)

    # If H < ph or W < pw, we must pad to at least patch size.
    H_needed = max(H, ph)
    W_needed = max(W, pw)

    # If positions were computed with H < ph, we already have [0]; still need pad to ph.
    pad_bottom = H_needed - H
    pad_right = W_needed - W
    pad_top = 0
    pad_left = 0

    # Apply padding bottom/right only (keeps coordinates simple)
    if pad_bottom > 0 or pad_right > 0:
        if pad_mode == "constant":
            x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode, value=pad_value)
        else:
            x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    else:
        x_pad = x

    _, _, H_pad, W_pad = x_pad.shape

    # Recompute positions on padded shape to guarantee coverage
    top_positions = _compute_positions(H_pad, ph, stride_h)
    left_positions = _compute_positions(W_pad, pw, stride_w)

    coords: List[Tuple[int, int]] = []
    patches_list: List[torch.Tensor] = []

    # Extract patches aligned across channels (slice neighbors)
    for top in top_positions:
        for left in left_positions:
            # [N, C, ph, pw]
            patch = x_pad[:, :, top : top + ph, left : left + pw]
            patches_list.append(patch)
            coords.append((top, left))

    # Stack into [P, N, C, ph, pw] then permute to [N, P, C, ph, pw]
    patches = torch.stack(patches_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()

    meta = {
        "H_in": H,
        "W_in": W,
        "H_pad": H_pad,
        "W_pad": W_pad,
        "ph": ph,
        "pw": pw,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "n_rows": len(top_positions),
        "n_cols": len(left_positions),
        "P": len(coords),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
    }

    return patches, (coords if return_coords else None), meta


def _make_blend_window(
    ph: int,
    pw: int,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create a 2D blending window of shape [1, 1, ph, pw] for overlap-add stitching.
    - "uniform": all ones
    - "hann"/"cosine": raised-cosine (Hann) in both dims, outer product -> 2D
    """
    if window == "uniform":
        w2d = torch.ones((ph, pw), device=device, dtype=dtype)
    elif window in ("hann", "cosine"):
        # Hann window in PyTorch is periodic by default; set periodic=False for overlap-add style.
        wh = torch.hann_window(ph, periodic=False, device=device, dtype=dtype).clamp_min(1e-6)
        ww = torch.hann_window(pw, periodic=False, device=device, dtype=dtype).clamp_min(1e-6)
        w2d = wh[:, None] * ww[None, :]
    else:
        raise ValueError("window must be one of: 'uniform', 'hann', 'cosine'")
    return w2d[None, None, :, :]  # [1,1,ph,pw]


def stitch_sliding_window_patches_core(
    patch_outputs: torch.Tensor,
    coords: List[Tuple[int, int]],
    meta: Dict[str, int],
    *,
    output_size: Optional[Tuple[int, int]] = None,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Stitch model outputs from sliding-window patches back into a full image using overlap-add.

    Typical workflow:
      patches, coords, meta = extract_sliding_window_patches_25d(...)
      logits_patches = model(patches_reshaped)  # then reshape back to [N,P,K,ph,pw]
      full_logits = stitch_sliding_window_patches(logits_patches, coords, meta)

    Args:
        patch_outputs:
            Either:
              - [N, P, K, ph, pw]  (recommended)
              - [N*P, K, ph, pw]   (will be reshaped using P from meta)
            where:
              N = number of 2.5D stacks / samples
              P = number of patches per sample
              K = channels (e.g., logits for num_classes)
              (ph, pw) = patch spatial size

        coords:
            List of (top, left) coordinates of length P, matching the extraction order.

        meta:
            Dict returned by extract_sliding_window_patches_2p5d, containing:
              H_in, W_in, H_pad, W_pad, ph, pw, P, pad_bottom, pad_right, etc.

        output_size:
            If provided, crop final result to (H, W). If None, uses (H_in, W_in).

        window:
            Blending window for overlap regions:
              - "uniform": simple average
              - "hann"/"cosine": smooth blending (recommended)

        eps:
            Small constant to avoid divide-by-zero.

    Returns:
        full: [N, K, H_out, W_out] stitched output (cropped to original size by default).
    """
    if patch_outputs.dim() == 4:
        # [N*P, K, ph, pw] -> [N, P, K, ph, pw]
        NP, K, ph, pw = patch_outputs.shape
        P = int(meta["P"])
        if NP % P != 0:
            raise ValueError(f"Cannot reshape: NP={NP} not divisible by P={P}.")
        N = NP // P
        patch_outputs = patch_outputs.view(N, P, K, ph, pw)
    elif patch_outputs.dim() == 5:
        # [N, P, K, ph, pw]
        N, P, K, ph, pw = patch_outputs.shape
        if P != int(meta["P"]):
            raise ValueError(f"P mismatch: patch_outputs has P={P}, meta['P']={meta['P']}.")
    else:
        raise ValueError(f"Expected patch_outputs dim 4 or 5, got {patch_outputs.dim()}.")

    if len(coords) != P:
        raise ValueError(f"coords length {len(coords)} != P {P}.")

    H_pad = int(meta["H_pad"])
    W_pad = int(meta["W_pad"])

    # Accumulators
    device = patch_outputs.device
    dtype = patch_outputs.dtype
    acc = torch.zeros((N, K, H_pad, W_pad), device=device, dtype=dtype)
    wacc = torch.zeros((N, 1, H_pad, W_pad), device=device, dtype=dtype)

    wpatch = _make_blend_window(ph, pw, window=window, device=device, dtype=dtype)  # [1,1,ph,pw]

    # Overlap-add accumulation
    # patch_outputs: [N, P, K, ph, pw]
    for p, (top, left) in enumerate(coords):
        patch = patch_outputs[:, p, :, :, :]          # [N, K, ph, pw]
        acc[:, :, top:top + ph, left:left + pw] += patch * wpatch
        wacc[:, :, top:top + ph, left:left + pw] += wpatch

    full = acc / (wacc + eps)  # [N, K, H_pad, W_pad]

    # Crop padding back to original size (or a user-specified output_size)
    if output_size is None:
        H_out, W_out = int(meta["H_in"]), int(meta["W_in"])
    else:
        H_out, W_out = output_size

    full = full[:, :, :H_out, :W_out].contiguous()
    return full


def stitch_sliding_window_patches(
    patch_outputs: torch.Tensor,
    coords: List[Tuple[int, int]],
    meta: Dict[str, int],
    *,
    output_size: Optional[Tuple[int, int]] = None,
    window: Literal["uniform", "hann", "cosine"] = "hann",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Stitch regression outputs from sliding-window patches back into a full image using overlap-add.

    This is identical in spirit to the segmentation/logits stitcher, but intended for
    regression targets (e.g., predicting the middle slice in a 2.5D setup).

    Args:
        patch_outputs:
            Either:
              - [N, P, K, ph, pw]  (recommended; K typically = 1 for regression)
              - [N*P, K, ph, pw]   (will be reshaped using P from meta)

            Note: K may be 1, but we keep it general.

        coords:
            List of (top, left) coordinates of length P, matching the extraction order.

        meta:
            Dict returned by extract_sliding_window_patches_2p5d, containing:
              H_in, W_in, H_pad, W_pad, ph, pw, P, pad_bottom, pad_right, etc.

        output_size:
            If provided, crop final result to (H, W). If None, uses (H_in, W_in).

        window:
            Blending window for overlap regions:
              - "uniform": simple average
              - "hann"/"cosine": smooth blending (recommended)

        eps:
            Small constant to avoid divide-by-zero.

    Returns:
        full: [N, K, H_out, W_out] stitched regression output (cropped to original size by default).
              For your middle-slice regression use-case, K=1 so output is [N, 1, H, W].
    """
    # Reuse the exact same stitcher as segmentation: it's channel-agnostic.
    return stitch_sliding_window_patches_core(
        patch_outputs=patch_outputs,
        coords=coords,
        meta=meta,
        output_size=output_size,
        window=window,
        eps=eps,
    )


class InferenceBatchSizeOptimizer:
    """
    Class for determining the optimal batch size to be used for inferencing
        -Differences in GPU memory (32GB V100 vs. 80GB A100), model size, and reconstructed image size can all influence
        how many images can be processed during inference. While we could process 1 image per batch, this is slow and wasteful.
        -This class helps determine the optimal size to be used
    params
        -model (obj) pytorch model to be used for inference
        -input_shape (tuple) size of the images to be denoised
        -device (obj) cuda device
        -max_batch_size (int) maximum batch size to check
        -precision (str) whether to use flaoting point 32 or amp
    """
    def __init__(self, model: nn.Module, input_shape: tuple, device: torch.device = torch.device('cuda'), 
                 max_batch_size: int = 512, precision: str = 'fp32'):
        self.model = model.eval().to(device)
        self.input_shape = input_shape  # (C, H, W) or (C, D, H, W) for 3D
        self.device = device
        self.max_batch_size = max_batch_size
        self.precision = precision.lower()

        if self.precision not in ['fp32', 'amp']:
            raise ValueError("precision must be either 'fp32' or 'amp'")

        self.cached_optimal_batch_size = None

    def get_available_memory(self):
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(self.device.index)[0] / 1024**2  # MB

    def estimate_peak_memory(self, batch_size: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        dummy_input = torch.randn((batch_size, 5, *self.input_shape), device=self.device)
        try:
            with torch.no_grad():
                if self.precision == 'amp':
                    with torch.autocast(device_type='cuda'):
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
        except RuntimeError as e:
            raise RuntimeError(f"OOM or other error at batch size {batch_size}: {e}")

        peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        return peak_mem

    def find_optimal_batch_size(self) -> int:
        if self.cached_optimal_batch_size is not None:
            return self.cached_optimal_batch_size

        low, high = 1, self.max_batch_size
        best = 1

        while low <= high:
            mid = (low + high) // 2
            try:
                _ = self.estimate_peak_memory(mid)
                best = mid
                low = mid + 1
            except RuntimeError:
                high = mid - 1

        self.cached_optimal_batch_size = best
        return best

    def profile(self):
        batch_size = self.find_optimal_batch_size()
        peak_memory = self.estimate_peak_memory(batch_size)
        available_memory = self.get_available_memory()
        
        return {
            'optimal_batch_size': batch_size,
            'peak_memory_used_MB': peak_memory,
            'available_memory_MB': available_memory
        }
