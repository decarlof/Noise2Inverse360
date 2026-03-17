
import torch
import torch.nn as nn
import torch.nn.functional as F


def laplacian_batch(x):
    # x: [B, C, H, W] (assumes grayscale: C=1)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # for grouped conv
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

class LCL(nn.Module):
    def __init__(self, ):
        super(LCL, self).__init__()

    def forward(self, pred):
        L = torch.abs(laplacian_batch(pred))
        threshold = torch.quantile(L, .80)
        edge_mask = (L > threshold)
        flat_mask = ~edge_mask

        edge_mean = L[edge_mask].mean() if edge_mask.any() else 0.0
        flat_mean = L[flat_mask].mean() if flat_mask.any() else 1e-6

        # Encourage this ratio to grow (i.e., minimize the inverse)
        return flat_mean / (edge_mean + 1e-6)
    

def laplacian_entropy_map(lap, bins = 256):
    # Compute entropy for each image in batch
    B = lap.shape[0]
    entropies = []
    for i in range(B):
        hist = torch.histc(lap[i, 0], bins=bins, min=0, max=lap[i, 0].max().item())
        hist = hist / hist.sum()
        hist = hist + 1e-8  # avoid log(0)
        entropy = -torch.sum(hist * torch.log(hist))
        entropies.append(entropy.item())
    return torch.tensor(entropies, device=lap.device)
