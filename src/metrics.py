import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


def compute_bpp_from_latent(z_hat, input_shape):
    _, c, h_z, w_z = z_hat.shape
    _, _, h, w = input_shape
    return (c * h_z * w_z) / (h * w)


def compute_psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def compute_quality_metrics(x, x_hat):
    """
    1枚分の復元品質（MSE，PSNR，SSIM，MS-SSIM）を計算する．

    Parameters
    ----------
    x : torch.Tensor
        原画像．shape は (1, C, H, W)，値域は [0, 1]
    x_hat : torch.Tensor
        復元画像．shape と値域は x と同じ

    Returns
    -------
    dict
        mse, psnr, ssim, ms_ssim
    """
    mse = F.mse_loss(x_hat, x, reduction="mean").item()
    psnr = compute_psnr_from_mse(mse)
    ssim_val = ssim(x_hat, x, data_range=1.0, size_average=True, win_size=7).item()
    ms_ssim_val = ms_ssim(x_hat, x, data_range=1.0, size_average=True, win_size=7).item()

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val,
        "ms_ssim": ms_ssim_val,
    }
