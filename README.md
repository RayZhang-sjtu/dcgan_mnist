# DCGAN for MNIST

> 🎨 一个严格对齐DCGAN论文的生成对抗网络实现，用于在MNIST数据集上生成手写数字图像

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 目录

- [项目简介](#项目简介)
- [GAN原理：天才伪造者与火眼金睛鉴定师](#gan原理天才伪造者与火眼金睛鉴定师)
- [项目亮点](#项目亮点)
- [可视化成果](#可视化成果)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练参数](#训练参数)
- [技术细节](#技术细节)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 📖 项目简介

**DCGAN (Deep Convolutional Generative Adversarial Networks)** 是2016年发表在ICLR的经典论文，它首次将卷积神经网络成功应用于生成对抗网络，开创了深度学习生成模型的新纪元。本项目是对DCGAN论文在MNIST数据集上的严格复现与改进。

### 为什么选择DCGAN？

DCGAN不仅是一篇里程碑式的论文，更是理解生成对抗网络的绝佳起点：

- **历史地位**：DCGAN是GAN领域的奠基之作，后续的Progressive GAN、StyleGAN等模型都深受其影响
- **架构清晰**：相比原始GAN，DCGAN使用卷积操作，结构更加直观易懂
- **效果卓越**：在MNIST、CelebA等数据集上都能生成高质量图像
- **教育价值**：代码简洁，是学习GAN原理的最佳实践项目

---

## 🎭 GAN原理：天才伪造者与火眼金睛鉴定师

想象这样一个场景：有一位**天才伪造者（生成器）**和一位**火眼金睛的鉴定师（判别器）**在互相博弈：

### 第一阶段：伪造者初出茅庐

- **生成器**：我画一幅"数字1"，你看看像不像真的？
- **判别器**：你这画的什么鬼，一看就是假的！（给出低分）

### 第二阶段：伪造者改进技艺

- **生成器**：好，我回去改进，这次画得更好！
- **判别器**：还是假的，不过比上次好一点了
- **生成器**：继续改进...
- **判别器**：嗯...这次有点像真的了，但还是有些瑕疵

### 第三阶段：双方都变强了

- **判别器**：我要变得更专业，不能总被欺骗！
- **生成器**：我要画得更逼真，让鉴定师都分不清真假！
- **两者持续博弈**，最终达到纳什均衡：生成器画的数字与真实数字几乎无法区分

### 这就是GAN的核心思想

```
生成器 (Generator) ──→ 生成假图像 ──→ 判别器 (Discriminator)
         ↑                                    ↓
         └─────────── 反馈（真/假） ←──────────┘
```

**生成器（Generator）**的目标是：让判别器无法区分生成图像和真实图像  
**判别器（Discriminator）**的目标是：准确识别哪些是真实图像，哪些是生成的假图像

这种对抗训练的过程，让生成器不断提升"造假"能力，最终生成逼真的图像！

---

## ✨ 项目亮点

### 🔍 严格的代码审计与对齐

本项目经过详细的代码审计，确保与DCGAN论文严格对齐：

#### 1. **激活函数修正**（论文对齐）

**发现问题**：初始实现使用了 `LeakyReLU(0.2)`  
**论文要求**：生成器隐藏层应使用 `ReLU`，只有判别器使用 `LeakyReLU`  
**修复方案**：将生成器的所有隐藏层激活函数改为 `ReLU(inplace=True)`

```python
# 修复前 ❌
nn.LeakyReLU(0.2, inplace=True)

# 修复后 ✅
nn.ReLU(inplace=True)
```

**影响**：ReLU激活函数更符合论文原始设计，确保复现的准确性

#### 2. **卷积核尺寸优化**（避免棋盘格伪影）

**发现问题**：Layer 2 使用了 `kernel_size=3, stride=2`，可能导致棋盘格伪影  
**论文标准**：DCGAN标准配置为 `kernel_size=4, stride=2, padding=1`  
**修复方案**：将Layer 2改为标准配置，并相应调整Layer 3的padding参数

```python
# 修复前 ❌
nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)  # 输出 7×7

# 修复后 ✅
nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 输出 8×8
```

**影响**：避免了转置卷积导致的棋盘格伪影，生成图像更加平滑自然

#### 3. **BatchNorm配置优化**（MNIST特化）

**配置**：所有BatchNorm层显式设置 `affine=False`

```python
nn.BatchNorm2d(channels, affine=False)
```

**说明**：对于MNIST这种简单的数据集，不使用可学习的缩放和平移参数（affine=False）可以减少过拟合风险，使训练更加稳定。

### 📊 完整的可视化系统

1. **潜在空间插值**：展示数字在潜空间中的平滑变换（如1→7的渐变过程）
2. **训练过程可视化**：记录每个epoch的生成结果，观察模型从"噪点"到"数字"的学习过程
3. **训练数据导出**：自动导出CSV/JSON格式的训练数据，方便后续分析

### 🎯 代码质量保证

- ✅ 模块化设计，结构清晰
- ✅ 完整的类型提示和文档字符串
- ✅ 符合PEP 8代码规范
- ✅ 完善的错误处理机制

---

## 🖼️ 可视化成果

### 1. 潜在空间插值（Latent Space Interpolation）

**什么是潜在空间插值？**

想象一下，如果我们有两个数字的"DNA"（潜在向量），比如数字1和数字7。潜在空间插值就是在这两个DNA之间生成一系列中间状态，观察数字是如何"平滑变身"的。

```
数字1 → [中间状态1] → [中间状态2] → ... → 数字7
```

这种平滑的变换证明了生成器学习到了有意义的潜在空间表示，而不是简单的记忆。这正是**流形学习（Manifold Learning）**的核心思想：相似的数字在潜在空间中距离更近，我们可以通过插值在数字之间平滑过渡。

**查看插值结果**：`results/interpolation_epoch_50.png`

### 2. 训练过程可视化

观察模型是如何从随机噪声逐步学会生成真实数字的：

- **Epoch 1**：生成的是模糊的噪点，几乎看不出数字形状
- **Epoch 10**：开始有数字的轮廓，但很模糊
- **Epoch 25**：数字形状清晰，但细节不够
- **Epoch 50**：生成的数字清晰、逼真，与真实MNIST数字难以区分

**查看训练过程**：`results/epoch_*_final.png`

### 3. 训练曲线分析

训练数据已自动导出为CSV格式（`results/training_data.csv`），包含：
- `D_loss`：判别器损失（越小越好）
- `G_loss`：生成器损失（越小越好）
- `D(x)`：真实图像的判别分数（接近1.0表示判别器能识别真实图像）
- `D(G(z))`：生成图像的判别分数（接近0.5表示训练平衡）

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- PyTorch 2.4+ (推荐使用CUDA版本以加速训练)
- 其他依赖见 `requirements.txt`

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/dcgan_mnist.git
cd dcgan_mnist
```

2. **创建虚拟环境**（推荐）
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 开始训练

**使用虚拟环境Python**（推荐）：
```bash
# Windows
.\venv\Scripts\python.exe train.py --n_epochs 50

# Linux/Mac
./venv/bin/python train.py --n_epochs 50
```

**或使用提供的脚本**：
```bash
# Windows PowerShell
.\run_train.ps1 --n_epochs 50

# Windows CMD
run_train.bat --n_epochs 50
```

**查看训练日志**：
```bash
tensorboard --logdir=./logs
```
然后在浏览器打开 http://localhost:6006

### 使用训练好的模型生成图像

模型权重保存在 `weights/` 目录中。你可以加载模型并生成图像：

```python
import torch
from models.generator import Generator
from utils.visualize import generate_fixed_noise_grid

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load('weights/generator_final.pth', map_location=device))
netG.eval()

# 生成图像
noise = torch.randn(64, 100, 1, 1, device=device)
generate_fixed_noise_grid(netG, noise, 'my_generated_images.png')
```

---

## 📁 项目结构

```
dcgan_mnist/
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── generator.py            # 生成器（Generator）
│   └── discriminator.py        # 判别器（Discriminator）
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── data_loader.py          # MNIST数据加载
│   ├── visualization.py        # 基础可视化工具
│   └── visualize.py            # 增强可视化工具（插值、网格等）
│
├── weights/                     # 模型权重（训练好的模型）
│   ├── generator_final.pth
│   └── discriminator_final.pth
│
├── results/                     # 训练结果
│   ├── interpolation_epoch_*.png   # 潜在空间插值结果
│   ├── epoch_*_final.png           # 每轮训练最终生成图像
│   ├── training_data.csv           # 训练数据（CSV格式）
│   └── training_data.json          # 训练数据（JSON格式）
│
├── logs/                        # TensorBoard日志
│   └── events.out.tfevents.*
│
├── data/                        # MNIST数据集（自动下载）
│   └── MNIST/
│
├── train.py                     # 主训练脚本
├── requirements.txt             # Python依赖
├── README.md                    # 本文件
├── .gitignore                   # Git忽略配置
├── run_train.ps1                # PowerShell训练脚本
└── run_train.bat                # 批处理训练脚本
```

---

## ⚙️ 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 128 | 批量大小 |
| `n_epochs` | 50 | 训练轮数 |
| `lr` | 0.0002 | 学习率（Adam优化器） |
| `beta1` | 0.5 | Adam优化器的beta1参数 |
| `nz` | 100 | 噪声向量维度 |
| `ngf` | 64 | 生成器基础特征数 |
| `ndf` | 64 | 判别器基础特征数 |
| `save_interval` | 10 | 每N轮保存一次模型 |
| `sample_interval` | 500 | 每N批次生成一次样本 |

**自定义参数示例**：
```bash
python train.py --batch_size 64 --n_epochs 100 --lr 0.0001
```

查看所有参数：
```bash
python train.py --help
```

---

## 🔧 技术细节

### 网络架构

#### 生成器 (Generator)

```
输入: 100维随机噪声 (1×1)
  ↓
ConvTranspose2d(100 → 512, 4×4, stride=1) + BatchNorm + ReLU  →  4×4
  ↓
ConvTranspose2d(512 → 256, 4×4, stride=2) + BatchNorm + ReLU  →  8×8
  ↓
ConvTranspose2d(256 → 128, 4×4, stride=2) + BatchNorm + ReLU  →  14×14
  ↓
ConvTranspose2d(128 → 1, 4×4, stride=2) + Tanh                →  28×28
```

**关键特点**：
- 使用转置卷积（ConvTranspose2d）进行上采样
- 隐藏层使用ReLU激活函数（论文要求）
- 输出层使用Tanh激活函数（输出范围[-1, 1]）
- 所有BatchNorm设置 `affine=False`（MNIST特化）

#### 判别器 (Discriminator)

```
输入: 28×28 灰度图像
  ↓
Conv2d(1 → 64, 4×4, stride=2) + LeakyReLU(0.2)              →  14×14
  ↓
Conv2d(64 → 128, 4×4, stride=2) + BatchNorm + LeakyReLU(0.2) →  7×7
  ↓
Conv2d(128 → 256, 3×3, stride=2) + BatchNorm + LeakyReLU(0.2) →  4×4
  ↓
Conv2d(256 → 1, 4×4, stride=1) + Sigmoid                    →  1×1
```

**关键特点**：
- 使用标准卷积进行下采样
- 所有隐藏层使用LeakyReLU(0.2)激活函数
- 输出层使用Sigmoid激活函数（输出概率值）

### 损失函数

使用二元交叉熵损失（BCELoss）：

- **判别器损失**：`L_D = -log(D(x)) - log(1 - D(G(z)))`
  - 最大化真实图像和生成图像的判别准确率
  
- **生成器损失**：`L_G = -log(D(G(z)))`
  - 最大化生成图像被判别为真实的概率

### 训练策略

- **交替训练**：先训练判别器，再训练生成器
- **优化器**：Adam (lr=0.0002, beta1=0.5, beta2=0.999)
- **权重初始化**：从N(0, 0.02)正态分布初始化卷积层权重
- **数据预处理**：将图像归一化到[-1, 1]范围（匹配Tanh输出）

---

## 📈 性能指标

### 训练时间（参考）

- **GPU (NVIDIA RTX 3050 Ti)**：约50分钟（50 epochs）
- **CPU**：约3-4小时（50 epochs）

### 生成质量

经过50轮训练后，生成的MNIST数字：
- ✅ 清晰度：与真实MNIST数字难以区分
- ✅ 多样性：能生成0-9所有数字
- ✅ 稳定性：训练过程稳定，无明显模式崩塌

---

## 🎓 学习资源

如果你想深入学习GAN和DCGAN：

1. **原始论文**：
   - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) (DCGAN, 2016)
   - [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) (原始GAN, 2014)

2. **优秀教程**：
   - [GAN by Ian Goodfellow](https://www.youtube.com/watch?v=HGYYEUSm-0Q)
   - [CS231n: Generative Models](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)

3. **相关项目**：
   - [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
   - [DCGAN TensorFlow Implementation](https://github.com/carpedm20/DCGAN-tensorflow)

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 确保代码通过基本的测试

---

## 📝 更新日志

### v1.0.0 (2025-12-22)

- ✨ 初始版本发布
- ✅ 严格对齐DCGAN论文实现
- ✅ 代码审计与修正（激活函数、卷积核配置）
- ✅ 完整的可视化系统（插值、训练过程）
- ✅ 训练数据自动导出（CSV/JSON）

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢DCGAN论文作者（Alec Radford, Luke Metz, Soumith Chintala）的开创性工作
- 感谢PyTorch团队提供优秀的深度学习框架
- 感谢所有为开源社区贡献的开发者

---

## 📧 联系方式

如有问题或建议，欢迎：
- 提交 [Issue](https://github.com/yourusername/dcgan_mnist/issues)
- 发送邮件至：your.email@example.com

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
