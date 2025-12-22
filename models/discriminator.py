"""
DCGAN判别器模型定义
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN判别器
    判断输入图像是真实图像还是生成图像
    使用卷积层将28x28图像下采样，输出一个标量（真实/虚假概率）
    """
    
    def __init__(self, nc=1, ndf=64):
        """
        参数:
            nc (int): 输入图像的通道数，MNIST为1（灰度图）
            ndf (int): 判别器特征图数量，默认64
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        # 判别器网络结构：逐步下采样图像尺寸
        self.main = nn.Sequential(
            # 输入: (batch_size, nc, 28, 28)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (batch_size, ndf, 14, 14)
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (batch_size, ndf*2, 7, 7)
            
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (batch_size, ndf*4, 4, 4)
            
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: (batch_size, 1, 1, 1)
        )
        
    def forward(self, input):
        """
        前向传播
        
        参数:
            input (torch.Tensor): 输入图像，形状为 [batch_size, nc, 28, 28]
        
        返回:
            output (torch.Tensor): 判别结果，形状为 [batch_size, 1, 1, 1]
                                  (在训练时会view成[batch_size, 1])
        """
        output = self.main(input)
        # 将输出从 [batch_size, 1, 1, 1] 展平为 [batch_size, 1]
        return output.view(output.size(0), -1)


def weights_init(m):
    """
    自定义权重初始化函数（与生成器共享）
    注意：由于BatchNorm的affine=False，不需要初始化weight和bias
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # 注意：当BatchNorm的affine=False时，没有可学习的weight和bias参数

