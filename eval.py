import numpy as np
import torch
import torch.nn.functional as F


def laplacian_batch(x):
    # x: [B, C, H, W] (assumes grayscale: C=1)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # for grouped conv
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

def laplacian_entropy_map(lap, bins = 256):
    # Compute entropy for each image in batch
    B = lap.shape[0]
    entropies = []
    for i in range(B):
        hist = torch.histc(lap[i, 0], bins=bins, min=0, max=lap[i, 0].max())
        hist = hist / hist.sum()
        hist = hist + 1e-8  # avoid log(0)
        entropy = -torch.sum(hist * torch.log(hist))
        entropies.append(entropy.item())
    return torch.tensor(entropies, device=lap.device)

def laplacian_score_batch(batch, entropy_thresh = .2, q = 0.9):
    # batch: [B, 1, H, W]
    lap = torch.abs(laplacian_batch(batch))  # [B, 1, H, W]
    B = lap.shape[0]
    scores = []

    entropies = laplacian_entropy_map(lap)
    
    for i in range(B):
        lap_i = lap[i, 0]  # [H, W]
        entropy = entropies[i]

        if entropy < entropy_thresh:
            # flat image: reward smoothness
            #score = -torch.mean(lap_i).item()
            smoothness = 1.0 / (lap_i.mean().item() + 1e-6)
            scores.append(smoothness)
        else:
            # compute threshold using quantile
            threshold = torch.quantile(lap_i, q)
            edge_mask = lap_i > threshold
            flat_mask = ~edge_mask

            edge_score = lap_i[edge_mask].mean() if edge_mask.any() else 0.0
            flat_score = lap_i[flat_mask].mean() if flat_mask.any() else 1e-6  # prevent div 0

            contrast = edge_score / (flat_score + 1e-6)
            scores.append(contrast)

    return float(np.mean(scores))