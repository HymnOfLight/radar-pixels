# SS-BiGAN: Spatial-Spectral Bi-directional GAN for Hyperspectral Nonlinear Unmixing

本项目基于论文《Looking into a Pixel by Nonlinear Unmixing -- A Generative
Approach》（Tang & Qi, 2026）中提出的“双向 GAN 非线性解混合”框架，进行的一项
方法学改进实现：**空间-光谱联合双向 GAN（SS-BiGAN）**。

与原文的逐像素处理不同，SS-BiGAN 通过 3D 卷积同时在空间和光谱两个维度上建模
邻域上下文，并保留了原始的“解混合 ↔ 混合”双向循环一致性约束，以数据驱动、无
模型的方式建模非线性光谱混合过程。

## 项目结构

```
.
├── README.md
├── requirements.txt
├── scripts/
│   └── download_datasets.sh        # 一键下载 Samson / Jasper Ridge / Urban
└── src/
    ├── __init__.py
    ├── data.py                     # 面片提取与 .mat 数据加载
    ├── losses.py                   # SAD / RMSE 等指标
    ├── models.py                   # 3D-CNN 解混合器、非线性混合器、判别器
    └── train.py                    # 训练主循环（TTUR + 标签平滑 + D/G 平衡）
```

## 核心思想

1. **空间-光谱 3D-CNN 解混合生成器 (`SpatialSpectralUnmixer`)**：输入
   `(Batch, 1, Bands, P, P)` 的局部面片，利用 `Conv3d` 同时聚合光谱与空间上下文，
   最后通过 `Softmax` 输出满足“非负性 (ANC)”和“和为一 (ASC)”约束的丰度向量。
2. **残差式非线性混合器 (`NonlinearMixer`)**：在线性混合模型 `A · E` 的基础上叠加
   一个幅值较小的 MLP 非线性残差。这种混合式架构能让网络在弱非线性场景下
   自然退化为线性模型，训练更稳定。
3. **光谱判别器 (`SpectralDiscriminator`)**：一个 MLP 判别真实光谱与重构光谱。
4. **双向循环损失**：
   - **正向流 (Unmix → Mix)**：真实光谱 → 丰度 → 重构光谱，用 MSE + SAD 监督；
   - **反向流 (Mix → Unmix)**：随机采样丰度 → 合成光谱 → 扩展为伪面片 → 解混合，
     用循环一致性 MSE 监督。
5. **训练平衡技巧**：TTUR（判别器学习率降为 10%）、标签平滑（0.9 / 0.1）、以及
   限制判别器更新频率（默认 G 每更新一次，D 每两次更新一次）。

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

### 2. 在模拟数据上冒烟测试

无需下载任何数据即可验证代码跑通：

```bash
python -m src.train --dataset mock --epochs 10 --log-every 2
```

### 3. 下载真实数据集

```bash
bash scripts/download_datasets.sh
```

脚本会创建 `datasets/Samson`、`datasets/JasperRidge`、`datasets/Urban` 三个子
目录，并下载对应的 `.mat` 文件（高光谱观测矩阵以及 Ground Truth 端元 / 丰度）。
下载后的目录结构类似：

```
datasets/
├── Samson/         Samson.mat         + Samson_GT.mat
├── JasperRidge/    jasperRidge2_R198.mat + Jasper_GT.mat
└── Urban/          Urban.mat          + end{4,5,6}_groundTruth.mat
```

下载源为 GitHub 上的公开学术镜像
[gaetanosettembre/data_unmixing](https://github.com/gaetanosettembre/data_unmixing)；
若未来某条 URL 失效，也可在同名 `scripts/download_datasets.sh` 里通过
`fetch_with_mirrors` 追加任意数量的镜像 URL。

### 4. 在真实数据上训练

```bash
python -m src.train --dataset samson --epochs 100
python -m src.train --dataset jasper --epochs 100
python -m src.train --dataset urban  --epochs 100 --batch-size 128
```

训练过程中会自动在每 `--log-every` 个 epoch 输出判别器与生成器的平均损失，
并在可用时计算丰度的 RMSE。训练完成后会将模型权重保存到
`checkpoints/ss_bigan_<dataset>.pt`。

## 常用命令行参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--dataset` | `mock` / `samson` / `jasper` / `urban` | `mock` |
| `--patch-size` | 空间面片窗口 `P` | `3` |
| `--batch-size` | 批大小 | `64` |
| `--epochs` | 训练轮数 | `100` |
| `--lr` | 生成器学习率 | `1e-4` |
| `--d-lr-scale` | 判别器学习率倍率（TTUR） | `0.1` |
| `--d-update-interval` | 判别器每 `N` 次迭代更新一次 | `2` |
| `--label-real` / `--label-fake` | 标签平滑 | `0.9` / `0.1` |
| `--w-forward` / `--w-backward` | 正 / 反向循环损失权重 | `10.0` / `5.0` |

## 数据集规格

| 数据集 | 空间尺寸 | 波段数 | 端元数 | 主要物质 |
| --- | --- | --- | --- | --- |
| Samson | 95 × 95 | 156 | 3 | Soil / Tree / Water |
| Jasper Ridge | 100 × 100 | 198 | 4 | Tree / Soil / Water / Road |
| Urban | 307 × 307 | 162 | 4–6 | Asphalt / Grass / Tree / Roof |

## 典型训练曲线

启用 TTUR + 标签平滑 + D/G 平衡之后，判别器损失会稳定在约 `0.55–0.68` 区间
（标签平滑下 BCE 的理论纳什均衡），生成器损失会在 `2.1–2.3` 之间收敛，不再
单调飙升。

## 许可证

本仓库仅用于学术研究与教学目的，所列第三方数据集版权归其原始作者所有。
