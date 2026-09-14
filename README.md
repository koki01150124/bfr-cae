# BFR-CAE

Bit-Flip Robust Compressive Autoencoder（BFR-CAE）の**推論・評価用**リポジトリです．

学習は行いません．学習済みモデルを読み込み，Kodak画像に対する推論，Bit Flip，復元品質の確認を行います．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/koki01150124/bfr-cae/blob/main/visualize_result.ipynb)

## 概要

画像圧縮モデルの二値潜在表現を伝送する際，ビット反転（Bit Flip）によって復元品質が大きく劣化する問題があります．BFR-CAE は，学習時にビット反転を考慮することで，伝送路ノイズに対して頑健な復元を目指す Compressive Autoencoder です．

本リポジトリでは，通常の CAE と BFR-CAE を同じ条件で比較できます．

## BFR-CAEの処理

```text
入力画像 x
  → Encoder
  → Binary Quantization（STE）
  → Bit Flip（推論時 BER = p）
  → Decoder
  → 復元画像 x_hat
```

1．Encoder が入力画像を連続値の潜在表現へ変換します．  
2．Binary Quantization により潜在表現を `{0, 1}` に量子化します．  
3．伝送を模擬し，各ビットを確率 `p`（BER）で反転します．  
4．Decoder が反転後の潜在表現から画像を復元します．

## CAEとの違い

| 項目 | CAE | BFR-CAE |
|---|---|---|
| 学習損失 | クリーン復元の MSE のみ | クリーン復元と Bit Flip 後復元の平均 MSE |
| 学習時 BER | なし | 各バッチで `[0, Train BER]` から一様サンプリング |
| ねらい | 高品質な通常圧縮 | ビット反転耐性を持たせた圧縮 |

推論時のモデル構造自体は共通で，どちらも Encoder → Binary Quantization →（任意で Bit Flip）→ Decoder です．違いは学習方法にあります．

## モデル構造

```text
入力画像 x (H×W×3)
    │
    ▼
 Encoder（stride-2 Conv ×3，ResBlock ×3）
    │  8倍ダウンサンプル
    ▼
 潜在表現 z  (H/8 × W/8 × C_z)  [0, 1] 連続値
    │
    ├─ Sigmoid → BinaryQuantize（STE）→ z_hat ∈ {0,1}
    │                                      │
    │                           BitFlipChannel(p) → z_tilde
    │                                      │
    ▼                                      ▼
 Decoder（PixelShuffle Subpix ×2，ResBlock ×3）
    │
    ▼
 復元画像（Clip STE で [0,1] にクランプ）
```

**bpp（bits per pixel）**

```text
bpp = C_z / 64
```

Encoder が空間を 8 倍ダウンサンプルし，二値量子化後は 1 ユニット = 1 bit のため，`C_z × (H/8) × (W/8) / (H × W) = C_z / 64` です．

公開デモの checkpoint は `C_z = 128`（bpp = 2.0）です．

## 推奨実行環境

**推奨: Google Colab**（GPU ランタイム）

- ブラウザから `visualize_result.ipynb` を開き，`Open in Colab` で実行できます．
- Colab にインストール済みの PyTorch を優先利用します（不要な再インストールは行いません）．
- GPU ランタイムが選択されていれば CUDA を自動利用し，なければ CPU にフォールバックします．

**ローカル動作確認環境**

- Python 3.12.12
- `uv` による仮想環境を推奨します
- CUDA や NVIDIA Driver の構築は必須ではありません（CPU でも実行可能です）

## Google Colabでの実行方法

1．次のボタンから Notebook を開きます．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/koki01150124/bfr-cae/blob/main/visualize_result.ipynb)

2．（任意）ランタイム → ランタイムのタイプを変更 → GPU を選択します．  
3．上から順にセルを実行します．  
4．Colab 上では，必要に応じて本リポジトリを clone し，不足パッケージのみをインストールします．

直接リンク:

```text
https://colab.research.google.com/github/koki01150124/bfr-cae/blob/main/visualize_result.ipynb
```

## ローカルでの実行方法

```bash
cd /path/to/bfr-cae
uv python install 3.12.12
uv venv --python 3.12.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter notebook visualize_result.ipynb
```

リポジトリルートで Notebook を開いてください．`src/` の import と相対パス参照のためです．

## 学習済みモデル

`checkpoints/` に公開デモ用の代表モデルのみを配置しています．

| ファイル | モデル | bpp | Train BER | その他 |
|---|---|---|---|---|
| `cae_bpp2.0_trainber0.10.pth` | CAE | 2.0 | —（Bit Flip なしで学習） | `latent_channels=128`，epoch=50，seed=0 |
| `bfr_cae_bpp2.0_trainber0.10.pth` | BFR-CAE | 2.0 | 0.10 | 同上．損失は average MSE |

両モデルは同一の bpp・データ・エポック数のもとで学習されており，比較条件を揃えています．研究条件が異なるモデル同士は比較しないでください．

## 推論方法

`visualize_result.ipynb` 先頭付近で次を指定します．

```python
IMAGE_INDEX = 1   # kodim01.png 〜 kodim24.png
TEST_BER = 0.01   # 推論時のビット反転確率
```

実行フロー:

```text
Kodak画像の選択
→ CAE / BFR-CAE のモデル読み込み
→ Encoder
→ Binary Quantization
→ Bit Flip
→ Decoder
→ 復元画像比較
→ PSNR，SSIM，MS-SSIM 表示
```

Kodak画像は 128×128 パッチ単位で推論します（研究用実装と同じ手順です）．

## 評価方法

復元画像に対し，次を計算します．

- **PSNR**: MSE から算出
- **SSIM**: `pytorch-msssim`，`win_size=7`，`data_range=1.0`
- **MS-SSIM**: 同上

実装は `src/metrics.py` にあり，研究用コードの評価方法を維持しています．

## 実験結果

Notebook を実行すると，同一 Kodak画像・同一 Test BER における CAE と BFR-CAE の復元画像，および PSNR / SSIM / MS-SSIM を並べて確認できます．

典型的な傾向:

- **BER = 0** では，CAE の方がやや高い復元品質になりやすいです．
- **BER > 0** では，BFR-CAE の方が品質低下が小さくなりやすいです．

正確な数値は実行環境と乱数（Bit Flip）に依存するため，Notebook 上の出力を正としてください．

## リポジトリ構成

```text
bfr-cae/
├── README.md
├── requirements.txt
├── .python-version
├── .gitignore
├── visualize_result.ipynb
├── checkpoints/
│   ├── cae_bpp2.0_trainber0.10.pth
│   └── bfr_cae_bpp2.0_trainber0.10.pth
├── data/
│   ├── kodim01.png
│   ├── ...
│   └── kodim24.png
└── src/
    ├── models/
    │   ├── __init__.py
    │   └── cae.py
    ├── bitflip.py
    ├── metrics.py
    └── utils.py
```

## 参考文献

- Theis, L., Shi, W., Cunningham, A., & Huszár, F. (2017). Lossy Image Compression with Compressive Autoencoders. *ICLR*.
- Kodak Lossless True Color Image Suite: <https://r0k.us/graphics/kodak/>

## ライセンス

ライセンスは選定中です．公開前に `LICENSE` を追加予定です．
