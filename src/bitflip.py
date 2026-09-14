import torch
import torch.nn as nn


class BitFlipChannel(nn.Module):
    def __init__(self, p=0.0):
        """
        Parameters
        ----------
        p : float
            ビット反転確率（BER）
            各ビットが p の確率で反転する
        """
        super().__init__()
        self.p = p

    def forward(self, z):
        """
        潜在表現 z にビット反転ノイズを付加する

        Parameters
        ----------
        z : torch.Tensor
            量子化後の二値潜在表現
            (0 または 1)

        Returns
        -------
        z_tilde : torch.Tensor
            ビット反転後の潜在表現
        """
        # もし，ビット反転 p が0未満なら 潜在表現 z をそのまま返す
        if self.p <= 0.0:
            return z

        # p で埋められた z と同じサイズのテンソルを作成し，ベルヌーイ分布を用いてビット反転する要素を決定
        flip = torch.bernoulli(torch.full_like(z, self.p))

        # ビット反転を実行
        #
        # 例:
        # z    = [0, 1, 1, 0]
        # flip = [0, 0, 1, 0]
        # 出力 = [0, 1, 0, 0]
        z_tilde = torch.abs(z - flip)
        return z_tilde
