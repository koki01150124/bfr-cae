import torch
import torch.nn as nn


class BinaryQuantize(torch.autograd.Function):
    """
    順伝播:
        x >= 0.5 なら 1，それ以外は 0 にする

    逆伝播:
        量子化の微分を無視して，勾配をそのまま前に流す（STE）
    """

    @staticmethod
    def forward(ctx, x):
        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantize(nn.Module):
    def forward(self, x):
        return BinaryQuantize.apply(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels)

    def forward(self, x):
        return self.norm(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class Subpix(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CAE(nn.Module):
    def __init__(self, latent_channels=96):
        super().__init__()

        def enc_res_block():
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, 128, 3, 1, 0),
                nn.LeakyReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, 128, 3, 1, 0),
            )

        def dec_res_block():
            return nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
            )

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 64, 5, 2, 0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 128, 5, 2, 0),
            nn.LeakyReLU(),
            Residual(enc_res_block()),
            Residual(enc_res_block()),
            Residual(enc_res_block()),
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, latent_channels, 5, 2, 0),
            ChannelLayerNorm(latent_channels),  # レイヤー正規化（Layer Normalization）
            nn.Sigmoid(),
        )

        self.quantize = Quantize()

        self.decoder = nn.Sequential(
            Subpix(latent_channels, 128),
            Residual(dec_res_block()),
            Residual(dec_res_block()),
            Residual(dec_res_block()),
            Subpix(128, 64),
            nn.Conv2d(64, 12, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_hat = self.quantize(z)
        x_hat = self.decoder(z_hat)
        x_hat = Clip.apply(x_hat)
        return x_hat, z, z_hat
