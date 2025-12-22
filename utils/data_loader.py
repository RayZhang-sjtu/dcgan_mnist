"""
MNIST数据集加载模块
自动下载MNIST数据集并进行预处理，支持批量加载
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size=128, num_workers=0, download=True):
    """
    获取MNIST数据集的DataLoader
    
    功能说明：
    1. 自动下载MNIST数据集到./data目录（如果不存在）
    2. 将图像转换为PyTorch张量
    3. 归一化到[-1, 1]范围（适合GAN训练）
    4. 返回训练和测试数据加载器
    
    参数:
        batch_size (int): 批量大小，默认128
        num_workers (int): 数据加载的线程数，Windows建议设为0，默认0
        download (bool): 如果数据不存在是否自动下载，默认True
    
    返回:
        train_loader (DataLoader): 训练数据加载器
        test_loader (DataLoader): 测试数据加载器
    
    示例:
        >>> train_loader, test_loader = get_mnist_loader(batch_size=64)
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # torch.Size([64, 1, 28, 28])
        ...     print(images.min(), images.max())  # tensor(-1.) tensor(1.)
    """
    # 设置数据存储路径
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据预处理流程
    # 步骤1: 转换为PyTorch张量 (0-255 -> 0.0-1.0)
    # 步骤2: 归一化到[-1, 1]范围，使用公式: normalized = (pixel / 255.0) * 2.0 - 1.0
    # 这样做的原因：GAN训练时，生成器输出tanh激活函数，范围是[-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转换为张量，并自动归一化到[0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]: (x - 0.5) / 0.5 = 2x - 1
    ])
    
    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    # 创建数据加载器
    # shuffle=True: 训练时打乱数据顺序，提高训练效果
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False  # GPU加速时使用固定内存
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print("正在加载MNIST数据集...")
    train_loader, test_loader = get_mnist_loader(batch_size=64)
    
    print(f"\n训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 获取一个批次的数据
    images, labels = next(iter(train_loader))
    print(f"\n批次数据形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"图像数值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"图像数据类型: {images.dtype}")

