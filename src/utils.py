import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.bitflip import BitFlipChannel
from src.models import CAE, Clip

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# 公開デモ用 checkpoint の既定設定（ex007_trainber_wide_sweep_bpp2, Train BER=0.10）
DEFAULT_LATENT_CHANNELS = 128
DEFAULT_BPP = DEFAULT_LATENT_CHANNELS / 64
DEFAULT_CAE_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "cae_bpp2.0_trainber0.10.pth"
DEFAULT_BFR_CAE_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "bfr_cae_bpp2.0_trainber0.10.pth"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=0):
    """
    seed設定
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True)


def kodak_path(image_index, data_dir=None):
    """
    IMAGE_INDEX（1〜24）に対応する Kodak 画像パスを返す．
    """
    if not 1 <= int(image_index) <= 24:
        raise ValueError(f"IMAGE_INDEX は 1〜24 の整数にしてください: {image_index}")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = root / f"kodim{int(image_index):02d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Kodak画像が見つかりません: {path}")
    return path


def load_kodak_image(image_index, data_dir=None, device=None):
    """
    Kodak画像を読み込み，形状 (1, C, H, W)，値域 [0, 1] のテンソルで返す．
    """
    path = kodak_path(image_index, data_dir=data_dir)
    img = Image.open(path).convert("RGB")
    x = T.ToTensor()(img).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    return x, path


def load_model(checkpoint_path, latent_channels=DEFAULT_LATENT_CHANNELS, device=None):
    """
    学習済み CAE / BFR-CAE の checkpoint を読み込む．
    """
    if device is None:
        device = get_device()

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint が見つかりません: {path}")

    model = CAE(latent_channels=latent_channels).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_with_patches(model, x, patch_size=128):
    """
    Patch推論（ビット反転なし）
    """
    model.eval()

    _, _, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0

    x_recon = torch.zeros_like(x)

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = x[:, :, i : i + patch_size, j : j + patch_size]

            z = model.encoder(patch)
            z_hat = model.quantize(z)
            patch_recon = model.decoder(z_hat)
            patch_recon = Clip.apply(patch_recon)

            x_recon[:, :, i : i + patch_size, j : j + patch_size] = patch_recon

    return x_recon


@torch.no_grad()
def reconstruct_with_patches_bitflip(model, x, p=0.0, patch_size=128):
    """
    Patch推論（ビット反転あり）
    """
    model.eval()
    device = x.device
    channel = BitFlipChannel(p).to(device)

    _, _, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0

    x_recon = torch.zeros_like(x)

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = x[:, :, i : i + patch_size, j : j + patch_size]

            z = model.encoder(patch)
            z_hat = model.quantize(z)
            z_tilde = channel(z_hat)

            patch_recon = model.decoder(z_tilde)
            patch_recon = Clip.apply(patch_recon)

            x_recon[:, :, i : i + patch_size, j : j + patch_size] = patch_recon

    return x_recon


@torch.no_grad()
def encode_quantize(model, x, patch_size=128):
    """
    Encoder → Binary Quantization までをパッチ単位で実行し，
    結合した z_hat を返す．
    """
    model.eval()
    _, _, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0

    # Encoder は空間を 8 倍ダウンサンプルする
    scale = 8
    z_hat_full = None

    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = x[:, :, i : i + patch_size, j : j + patch_size]
            z = model.encoder(patch)
            z_hat = model.quantize(z)

            if z_hat_full is None:
                _, c, _, _ = z_hat.shape
                z_hat_full = torch.zeros(
                    1, c, H // scale, W // scale, device=x.device, dtype=z_hat.dtype
                )

            zi = i // scale
            zj = j // scale
            zh = patch_size // scale
            zw = patch_size // scale
            z_hat_full[:, :, zi : zi + zh, zj : zj + zw] = z_hat

    return z_hat_full


def to_numpy_image(t):
    """
    (1, C, H, W) または (C, H, W) のテンソルを HWC の numpy 画像へ変換する．
    """
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
