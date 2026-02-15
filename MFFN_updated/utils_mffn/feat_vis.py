import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _pca3(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:3].T

@torch.no_grad()
def feat_to_rgb_pca(feat: torch.Tensor, out_hw=None, eps=1e-6) -> np.ndarray:
    if feat.dim() == 4:
        feat = feat[0]
    C, H, W = feat.shape

    x = feat.float()
    x = (x - x.mean(dim=(1,2), keepdim=True)) / (x.std(dim=(1,2), keepdim=True) + eps)

    X = x.permute(1,2,0).reshape(-1, C).cpu().numpy()
    X3 = _pca3(X)

    X3 = X3 - X3.min(axis=0, keepdims=True)
    X3 = X3 / (X3.max(axis=0, keepdims=True) + eps)

    rgb = X3.reshape(H, W, 3)

    if out_hw is not None and (out_hw[0] != H or out_hw[1] != W):
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()
        rgb_t = F.interpolate(rgb_t, size=out_hw, mode="bilinear", align_corners=False)
        rgb = rgb_t[0].permute(1,2,0).cpu().numpy()

    return (rgb * 255).clip(0, 255).astype(np.uint8)

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3,1,1)
    return (x * std + mean).clamp(0, 1)

@torch.no_grad()
def save_feat_grid(
    save_path: str,
    input_img_chw: torch.Tensor,
    feat_list: list,
    titles: list,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    H, W = input_img_chw.shape[1], input_img_chw.shape[2]
    inp = denorm_imagenet(input_img_chw).permute(1,2,0).detach().cpu().numpy()

    cols = 1 + len(feat_list)
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))

    ax[0].imshow(inp)
    ax[0].set_title("Input")
    ax[0].axis("off")

    for i, (feat, t) in enumerate(zip(feat_list, titles), start=1):
        rgb = feat_to_rgb_pca(feat, out_hw=(H, W))
        ax[i].imshow(rgb)
        ax[i].set_title(t)
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)