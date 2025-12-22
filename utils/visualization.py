"""
可视化工具模块
用于显示和保存生成的图像
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def save_images(images, filename, nrow=8, normalize=True, save_dir='./generated_images'):
    """
    保存图像网格
    
    参数:
        images (torch.Tensor): 图像张量，形状为 [N, C, H, W]
        filename (str): 保存的文件名
        nrow (int): 每行显示的图像数量
        normalize (bool): 是否将图像从[-1, 1]反归一化到[0, 1]
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量转换为numpy数组
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # 反归一化：从[-1, 1]恢复到[0, 1]
    if normalize:
        images = (images + 1) / 2.0
        images = np.clip(images, 0, 1)
    
    # 如果是单通道图像，去掉通道维度
    if images.shape[1] == 1:
        images = images.squeeze(1)
    
    # 创建图像网格
    n_images = images.shape[0]
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图像已保存到: {save_path}")


def show_images(images, nrow=8, normalize=True):
    """
    显示图像网格（不保存）
    
    参数:
        images (torch.Tensor): 图像张量，形状为 [N, C, H, W]
        nrow (int): 每行显示的图像数量
        normalize (bool): 是否将图像从[-1, 1]反归一化到[0, 1]
    """
    # 将张量转换为numpy数组
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # 反归一化
    if normalize:
        images = (images + 1) / 2.0
        images = np.clip(images, 0, 1)
    
    # 如果是单通道图像，去掉通道维度
    if images.shape[1] == 1:
        images = images.squeeze(1)
    
    n_images = images.shape[0]
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 2, nrow_actual * 2))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        axes[i].axis('off')
    
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

