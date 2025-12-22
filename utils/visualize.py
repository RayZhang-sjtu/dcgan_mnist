"""
增强的可视化工具模块
包含潜在空间插值、固定噪声网格生成等功能
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def latent_space_interpolation(netG, z1, z2, n_steps=10, device='cuda', normalize=True):
    """
    在潜在空间中进行线性插值，生成中间状态的图像
    
    参数:
        netG: 训练好的生成器模型
        z1, z2: 两个噪声向量，形状为 [1, nz, 1, 1] 或 [nz]
        n_steps: 插值步数，默认10
        device: 设备 ('cuda' 或 'cpu')
        normalize: 是否将图像从[-1, 1]反归一化到[0, 1]
    
    返回:
        images: 插值图像的张量，形状为 [n_steps, C, H, W]
    """
    netG.eval()
    
    # 确保z1和z2是正确的形状
    if len(z1.shape) == 1:
        z1 = z1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, nz, 1, 1]
    if len(z2.shape) == 1:
        z2 = z2.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    z1 = z1.to(device)
    z2 = z2.to(device)
    
    # 生成插值系数
    alphas = torch.linspace(0, 1, n_steps).to(device)
    
    interpolated_images = []
    with torch.no_grad():
        for alpha in alphas:
            # 线性插值
            z_interp = (1 - alpha) * z1 + alpha * z2
            # 生成图像
            img = netG(z_interp)
            interpolated_images.append(img.cpu())
    
    # 拼接所有图像
    images = torch.cat(interpolated_images, dim=0)
    
    # 反归一化
    if normalize:
        images = (images + 1) / 2.0
        images = torch.clamp(images, 0, 1)
    
    return images


def save_interpolation_grid(images, filename, save_dir='./results', nrow=None):
    """
    保存插值图像网格
    
    参数:
        images: 图像张量，形状为 [N, C, H, W]
        filename: 保存的文件名
        save_dir: 保存目录
        nrow: 每行显示的图像数量（None表示自动计算）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # 如果是单通道图像，去掉通道维度
    if images.shape[1] == 1:
        images = images.squeeze(1)
    
    n_images = images.shape[0]
    if nrow is None:
        nrow = n_images  # 默认一行显示所有图像
    
    ncol = min(nrow, n_images)
    nrow_actual = (n_images + ncol - 1) // ncol
    
    # 创建图像网格
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 1.5, nrow_actual * 1.5))
    if nrow_actual == 1:
        if ncol == 1:
            axes = [axes]
        else:
            axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        axes[i].axis('off')
        # 添加步数标签
        axes[i].set_title(f'Step {i}', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"插值图像已保存到: {save_path}")


def generate_fixed_noise_grid(netG, fixed_noise, filename, save_dir='./results', 
                               nrow=8, normalize=True):
    """
    使用固定噪声生成图像网格（用于制作训练演变动画）
    
    参数:
        netG: 生成器模型
        fixed_noise: 固定噪声向量，形状为 [N, nz, 1, 1]
        filename: 保存的文件名
        save_dir: 保存目录
        nrow: 每行显示的图像数量
        normalize: 是否反归一化
    """
    os.makedirs(save_dir, exist_ok=True)
    
    netG.eval()
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()
    
    # 反归一化
    if normalize:
        fake_images = (fake_images + 1) / 2.0
        fake_images = torch.clamp(fake_images, 0, 1)
    
    # 转换为numpy数组
    images = fake_images.numpy()
    
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
    print(f"固定噪声图像已保存到: {save_path}")


def create_interpolation_comparison(netG, z1, z2, n_steps=10, device='cuda', 
                                    save_path='./results/interpolation_comparison.png'):
    """
    创建插值对比图（水平排列）
    
    参数:
        netG: 生成器模型
        z1, z2: 两个噪声向量
        n_steps: 插值步数
        device: 设备
        save_path: 保存路径
    """
    images = latent_space_interpolation(netG, z1, z2, n_steps, device)
    save_interpolation_grid(images, os.path.basename(save_path), 
                           os.path.dirname(save_path), nrow=n_steps)


# 保留原有的save_images函数（向后兼容）
def save_images(images, filename, nrow=8, normalize=True, save_dir='./generated_images'):
    """
    保存图像网格（向后兼容原有函数）
    
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

