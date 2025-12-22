"""
DCGAN生成器模型定义
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN生成器
    将随机噪声向量转换为28x28的MNIST图像
    """

    def __init__(self, nz=100, ngf=64, nc=1):
        """
        参数:
            nz (int): 输入噪声向量的维度，默认100
            ngf (int): 生成器特征图数量，默认64
            nc (int): 输出图像的通道数，MNIST为1（灰度图）
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        # 生成器网络结构（按照DCGAN论文标准配置）
        # 修改：使用ReLU替代LeakyReLU，修复Layer 2的kernel_size以避免棋盘格伪影
        self.main = nn.Sequential(
            # Layer 1: 输入层
            # 输入: (batch_size, nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8, affine=False),
            nn.ReLU(inplace=True),
            # 状态: (batch_size, ngf*8, 4, 4)
            
            # Layer 2: 上采样层（修复：使用kernel=4, stride=2, padding=1标准配置）
            # 注意：输出尺寸从4x4变为8x8（而非7x7），需要调整后续层
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4, affine=False),
            nn.ReLU(inplace=True),
            # 状态: (batch_size, ngf*4, 8, 8)
            
            # Layer 3: 上采样层（调整：从8x8到14x14，使用kernel=4, stride=2, padding=2）
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 2, affine=False),
            nn.ReLU(inplace=True),
            # 状态: (batch_size, ngf*2, 14, 14)
            
            # Layer 4: 输出层
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: (batch_size, nc, 28, 28)
        )

    def forward(self, input):
        """
        前向传播

        参数:
            input (torch.Tensor): 噪声向量，形状为 [batch_size, nz, 1, 1]

        返回:
            output (torch.Tensor): 生成的图像，形状为 [batch_size, nc, 28, 28]
        """
        return self.main(input)


def weights_init(m):
    """
    自定义权重初始化函数
    注意：由于BatchNorm的affine=False，不需要初始化weight和bias
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # 注意：当BatchNorm的affine=False时，没有可学习的weight和bias参数