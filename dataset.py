"""
MNIST 数据集加载模块

MNIST（Modified National Institute of Standards and Technology）是一个经典的手写数字数据集：
- 训练集：60,000 张 28x28 灰度图像
- 测试集：10,000 张 28x28 灰度图像
- 类别：0-9 共 10 个数字

本模块负责：
1. 自动下载 MNIST 数据集（首次运行时）
2. 对图像进行预处理（转为张量 + 标准化）
3. 创建 DataLoader 用于批量加载数据
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64, data_dir='./data', num_workers=2):
    """
    获取 MNIST 训练集和测试集的 DataLoader

    参数:
        batch_size (int): 每个批次的样本数量，默认 64
            - 较小的 batch_size（如 32）：训练更稳定，但速度较慢
            - 较大的 batch_size（如 256）：训练更快，但可能不够稳定
        data_dir (str): 数据集存储路径，默认为当前目录下的 ./data/
        num_workers (int): 数据加载的子进程数量，默认 2
            - 设为 0 表示在主进程中加载（适合调试）

    返回:
        tuple: (train_loader, test_loader) 分别用于训练和测试

    使用示例:
        >>> train_loader, test_loader = get_data_loaders(batch_size=64)
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # torch.Size([64, 1, 28, 28])
        ...     print(labels.shape)  # torch.Size([64])
        ...     break
    """

    # ==================== 数据预处理（transforms） ====================
    # transforms.Compose 将多个预处理步骤串联起来，按顺序执行
    transform = transforms.Compose([
        # 步骤 1: 将 PIL 图像或 numpy 数组转换为 PyTorch 张量
        # - 像素值从 [0, 255] 的整数 → [0.0, 1.0] 的浮点数
        # - 数据形状从 (H, W) → (C, H, W)，即 (28, 28) → (1, 28, 28)
        transforms.ToTensor(),

        # 步骤 2: 标准化（Normalize）
        # 公式: output = (input - mean) / std
        # - mean=0.1307, std=0.3081 是 MNIST 数据集的全局统计值
        # - 标准化后数据大致分布在 [-0.42, 2.82] 范围内
        # - 这有助于加速模型训练的收敛
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ==================== 加载训练集 ====================
    train_dataset = datasets.MNIST(
        root=data_dir,      # 数据存储根目录
        train=True,          # True 表示加载训练集（60,000 张）
        download=True,       # 如果本地没有数据，自动从网上下载
        transform=transform  # 应用上面定义的预处理
    )

    # ==================== 加载测试集 ====================
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,         # False 表示加载测试集（10,000 张）
        download=True,
        transform=transform
    )

    # ==================== 创建 DataLoader ====================
    # DataLoader 负责将数据集按批次（batch）加载，并支持多进程和打乱顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # 训练时打乱数据顺序，避免模型学到数据的排列规律
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,       # 测试时不需要打乱，保持一致性
        num_workers=num_workers
    )

    return train_loader, test_loader


def show_samples(data_loader, num_samples=16):
    """
    可视化数据集中的样本图片

    参数:
        data_loader: DataLoader 对象
        num_samples (int): 显示的样本数量，默认 16

    说明:
        此函数会弹出一个 matplotlib 窗口，展示手写数字图片及其标签。
        如果在无图形界面的服务器上运行，图片会保存到 samples.png 文件。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # 设置 matplotlib 支持中文显示
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 从 DataLoader 中取一个批次的数据
    images, labels = next(iter(data_loader))

    # 限制显示数量
    num_samples = min(num_samples, len(images))

    # 计算网格布局（尽量接近正方形）
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_samples):
        # images[i] 的形状是 (1, 28, 28)，squeeze 去掉通道维度变成 (28, 28)
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'label: {labels[i].item()}')
        axes[i].axis('off')  # 隐藏坐标轴

    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # 尝试显示图片，如果无法显示则保存为文件
    try:
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            plt.savefig('samples.png', dpi=100)
            print("样本图片已保存到 samples.png")
    except Exception:
        plt.savefig('samples.png', dpi=100)
        print("样本图片已保存到 samples.png")
    finally:
        plt.close()


# ==================== 直接运行本文件可以测试数据加载 ====================
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders(batch_size=64)
    print(f"训练集大小: {len(train_loader.dataset)} 张图片")
    print(f"测试集大小: {len(test_loader.dataset)} 张图片")
    print(f"每个批次: {train_loader.batch_size} 张图片")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")

    # 查看一个批次的数据形状
    images, labels = next(iter(train_loader))
    print(f"\n一个批次的图片形状: {images.shape}")
    print(f"  - 批次大小: {images.shape[0]}")
    print(f"  - 通道数: {images.shape[1]}（灰度图只有 1 个通道）")
    print(f"  - 图片高度: {images.shape[2]} 像素")
    print(f"  - 图片宽度: {images.shape[3]} 像素")
    print(f"一个批次的标签形状: {labels.shape}")

    # 显示样本图片
    show_samples(train_loader)
