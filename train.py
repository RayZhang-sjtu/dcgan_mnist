"""
DCGAN训练主脚本
实现DCGAN的完整训练流程，包括交替训练生成器和判别器
"""

import os
import argparse
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.generator import Generator, weights_init as gen_weights_init
from models.discriminator import Discriminator, weights_init as disc_weights_init
from utils.data_loader import get_mnist_loader
from utils.visualization import save_images
from utils.visualize import latent_space_interpolation, save_interpolation_grid, generate_fixed_noise_grid


def train_dcgan(
    batch_size=128,
    nz=100,
    n_epochs=50,
    lr=0.0002,
    beta1=0.5,
    ngf=64,
    ndf=64,
    nc=1,
    save_interval=10,
    sample_interval=500,
    resume_epoch=None
):
    """
    DCGAN训练函数
    
    参数:
        batch_size (int): 批量大小，默认128
        nz (int): 噪声向量维度，默认100
        n_epochs (int): 训练轮数，默认50
        lr (float): 学习率，默认0.0002
        beta1 (float): Adam优化器的beta1参数，默认0.5
        ngf (int): 生成器特征图数量，默认64
        ndf (int): 判别器特征图数量，默认64
        nc (int): 图像通道数，MNIST为1，默认1
        save_interval (int): 每隔多少轮保存一次模型，默认10
        sample_interval (int): 每隔多少批次生成一次样本图像，默认500
        resume_epoch (int): 从第几轮恢复训练，None表示从头开始
    """
    # 设置设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./generated_images', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir='./logs')
    
    # 加载数据
    print("正在加载MNIST数据集...")
    train_loader, _ = get_mnist_loader(batch_size=batch_size, num_workers=0)
    print(f"数据集加载完成，共 {len(train_loader)} 个批次")
    
    # 初始化生成器
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netG.apply(gen_weights_init)
    
    # 初始化判别器
    netD = Discriminator(nc=nc, ndf=ndf).to(device)
    netD.apply(disc_weights_init)
    
    # 如果恢复训练，加载模型
    start_epoch = 0
    if resume_epoch is not None:
        print(f"从第 {resume_epoch} 轮恢复训练...")
        netG.load_state_dict(torch.load(f'./saved_models/generator_epoch_{resume_epoch}.pth'))
        netD.load_state_dict(torch.load(f'./saved_models/discriminator_epoch_{resume_epoch}.pth'))
        start_epoch = resume_epoch
    
    # 定义损失函数（二元交叉熵）
    criterion = nn.BCELoss()
    
    # 设置优化器（Adam，按照DCGAN论文推荐参数）
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # 固定噪声用于生成样本图像（便于观察训练过程）
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # 用于潜在空间插值的两个随机噪声向量
    z1 = torch.randn(1, nz, 1, 1, device=device)
    z2 = torch.randn(1, nz, 1, 1, device=device)
    
    # 用于存储训练数据的列表
    training_data = {
        'losses': {'D_loss': [], 'G_loss': []},
        'scores': {'D_x': [], 'D_G_z1': [], 'D_G_z2': []},
        'steps': []
    }
    
    # 训练标签
    real_label = 1.0
    fake_label = 0.0
    
    print("\n开始训练...")
    print(f"训练参数: batch_size={batch_size}, lr={lr}, n_epochs={n_epochs}")
    print("-" * 60)
    
    # 训练循环
    global_step = 0
    for epoch in range(start_epoch, n_epochs):
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # =============================
            # 训练判别器 (Discriminator)
            # =============================
            netD.zero_grad()
            
            # 1. 训练真实图像
            label = torch.full((batch_size_actual,), real_label, dtype=torch.float, device=device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()  # 真实图像的平均判别分数
            
            # 2. 训练生成图像
            noise = torch.randn(batch_size_actual, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()  # 生成图像的平均判别分数
            
            # 计算判别器总损失并更新
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # =============================
            # 训练生成器 (Generator)
            # =============================
            netG.zero_grad()
            label.fill_(real_label)  # 生成器希望判别器认为生成图像是真实的
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()  # 生成器训练后的判别分数
            optimizerG.step()
            
            # 记录损失到TensorBoard和训练数据
            if global_step % 50 == 0:
                writer.add_scalar('Loss/Discriminator', errD.item(), global_step)
                writer.add_scalar('Loss/Generator', errG.item(), global_step)
                writer.add_scalar('Score/D_x', D_x, global_step)  # 真实图像判别分数
                writer.add_scalar('Score/D_G_z1', D_G_z1, global_step)  # 生成图像判别分数（训练前）
                writer.add_scalar('Score/D_G_z2', D_G_z2, global_step)  # 生成图像判别分数（训练后）
                
                # 保存训练数据
                training_data['steps'].append(global_step)
                training_data['losses']['D_loss'].append(errD.item())
                training_data['losses']['G_loss'].append(errG.item())
                training_data['scores']['D_x'].append(D_x)
                training_data['scores']['D_G_z1'].append(D_G_z1)
                training_data['scores']['D_G_z2'].append(D_G_z2)
            
            # 更新进度条
            pbar.set_postfix({
                'D_loss': f'{errD.item():.4f}',
                'G_loss': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z2:.4f}'
            })
            
            # 定期生成样本图像（使用固定噪声网格）
            if global_step % sample_interval == 0:
                generate_fixed_noise_grid(
                    netG, fixed_noise, 
                    f'epoch_{epoch+1}_step_{global_step}.png',
                    save_dir='./results'
                )
            
            global_step += 1
        
        # 每个epoch结束后生成一次样本图像
        generate_fixed_noise_grid(
            netG, fixed_noise,
            f'epoch_{epoch+1}_final.png',
            save_dir='./results'
        )
        
        # 每个epoch结束后进行潜在空间插值
        try:
            netG.eval()
            interpolated_images = latent_space_interpolation(
                netG, z1.squeeze(), z2.squeeze(), 
                n_steps=10, device=device
            )
            save_interpolation_grid(
                interpolated_images,
                f'interpolation_epoch_{epoch+1}.png',
                save_dir='./results'
            )
            netG.train()
        except Exception as e:
            print(f"警告：潜在空间插值生成失败: {e}")
            netG.train()
        
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f'./saved_models/generator_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'./saved_models/discriminator_epoch_{epoch+1}.pth')
            print(f"\n模型已保存: epoch {epoch+1}")
        
        print()  # 空行分隔每个epoch
    
    # 训练结束，保存最终模型
    torch.save(netG.state_dict(), './saved_models/generator_final.pth')
    torch.save(netD.state_dict(), './saved_models/discriminator_final.pth')
    
    # 导出训练数据为JSON和CSV
    try:
        # 保存为JSON
        json_path = './results/training_data.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2)
        print(f"\n训练数据已保存为JSON: {json_path}")
        
        # 保存为CSV
        csv_path = './results/training_data.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer_csv = csv.writer(f)
            # 写入表头
            writer_csv.writerow(['step', 'D_loss', 'G_loss', 'D_x', 'D_G_z1', 'D_G_z2'])
            # 写入数据
            for i in range(len(training_data['steps'])):
                writer_csv.writerow([
                    training_data['steps'][i],
                    training_data['losses']['D_loss'][i],
                    training_data['losses']['G_loss'][i],
                    training_data['scores']['D_x'][i],
                    training_data['scores']['D_G_z1'][i],
                    training_data['scores']['D_G_z2'][i]
                ])
        print(f"训练数据已保存为CSV: {csv_path}")
    except Exception as e:
        print(f"警告：训练数据导出失败: {e}")
    
    print("\n训练完成！最终模型已保存。")
    print(f"TensorBoard日志保存在: ./logs")
    print(f"生成图像保存在: ./generated_images")
    print(f"可视化结果保存在: ./results")
    
    writer.close()


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='DCGAN训练脚本')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小 (默认: 128)')
    parser.add_argument('--nz', type=int, default=100, help='噪声向量维度 (默认: 100)')
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数 (默认: 50)')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率 (默认: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1参数 (默认: 0.5)')
    parser.add_argument('--ngf', type=int, default=64, help='生成器特征图数量 (默认: 64)')
    parser.add_argument('--ndf', type=int, default=64, help='判别器特征图数量 (默认: 64)')
    parser.add_argument('--nc', type=int, default=1, help='图像通道数 (默认: 1)')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔（轮数） (默认: 10)')
    parser.add_argument('--sample_interval', type=int, default=500, help='样本生成间隔（批次） (默认: 500)')
    parser.add_argument('--resume', type=int, default=None, help='从第几轮恢复训练 (默认: None)')
    
    args = parser.parse_args()
    
    train_dcgan(
        batch_size=args.batch_size,
        nz=args.nz,
        n_epochs=args.n_epochs,
        lr=args.lr,
        beta1=args.beta1,
        ngf=args.ngf,
        ndf=args.ndf,
        nc=args.nc,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        resume_epoch=args.resume
    )


if __name__ == '__main__':
    # 如果没有命令行参数，使用默认参数训练
    import sys
    if len(sys.argv) == 1:
        print("使用默认参数训练。可以使用 --help 查看所有可用参数。\n")
        train_dcgan()
    else:
        main()

