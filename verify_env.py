import time

import torch
import torchvision
import numpy
import pandas
import matplotlib
import seaborn
import tqdm
import tensorboard


def main() -> None:
    print("=== PyTorch & CUDA 信息 ===")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
    else:
        print("未检测到可用 CUDA，将在 CPU 上运行基准测试。")
        device = torch.device("cpu")

    print("\n=== 依赖库版本检查 ===")
    libs = [
        ("torchvision", torchvision.__version__),
        ("numpy", numpy.__version__),
        ("pandas", pandas.__version__),
        ("matplotlib", matplotlib.__version__),
        ("seaborn", seaborn.__version__),
        ("tqdm", tqdm.__version__),
        ("tensorboard", tensorboard.__version__),
    ]
    for name, version in libs:
        print(f"{name:12s}: OK (v{version})")

    print("\n=== 简单张量运算测试 ===")
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize() if cuda_available else None
    start = time.time()
    for _ in range(50):
        z = x @ y
        z = torch.relu(z)
    torch.cuda.synchronize() if cuda_available else None
    elapsed = time.time() - start
    print(f"矩阵乘 + ReLU 50 次耗时: {elapsed:.3f}s （设备: {device}）")


if __name__ == "__main__":
    main()


