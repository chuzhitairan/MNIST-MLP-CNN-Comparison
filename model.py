"""
MNIST 模型定义模块

本模块定义了两种用于手写数字识别的神经网络模型：

1. MLP（多层感知机）—— 最基础的全连接神经网络
   - 将 28x28 的图片展平为 784 维向量
   - 通过多个全连接层进行分类
   - 结构简单，适合入门理解神经网络的基本原理

2. CNN（卷积神经网络）—— LeNet 风格的经典卷积网络
   - 直接处理 28x28 的二维图像
   - 使用卷积层提取图像的空间特征（边缘、纹理等）
   - 准确率更高，是图像识别任务的主流方法

两种模型的对比：
┌────────┬──────────┬──────────┬───────────────┐
│  模型  │  参数量  │ 测试准确率│    特点       │
├────────┼──────────┼──────────┼───────────────┤
│  MLP   │  ~267K   │  ~98.0%  │ 结构简单易懂  │
│  CNN   │  ~600K   │  ~99.2%  │ 利用空间信息  │
└────────┴──────────┴──────────┴───────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  模型 1: MLP（多层感知机 / Multi-Layer Perceptron）
# ============================================================================

class MLP(nn.Module):
    """
    多层感知机（全连接神经网络）

    网络结构:
        输入 (784) → 全连接层 (256) → ReLU → Dropout
                   → 全连接层 (256) → ReLU → Dropout
                   → 输出层 (10)

    工作原理:
        1. 将 28x28 的图片"展平"成一个 784 维的向量
        2. 通过全连接层（Linear）进行线性变换：y = Wx + b
        3. 通过 ReLU 激活函数引入非线性：max(0, x)
        4. 通过 Dropout 随机丢弃一些神经元，防止过拟合
        5. 最终输出 10 个数值，代表属于每个数字(0-9)的"得分"
    """

    def __init__(self, input_dims=784, hidden_dims=None, num_classes=10):
        """
        参数:
            input_dims (int): 输入维度，MNIST 图片为 28*28=784
            hidden_dims (list): 每个隐藏层的神经元数量，默认 [256, 256]
            num_classes (int): 输出类别数，数字 0-9 共 10 类
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # ---- 构建网络层 ----
        # nn.ModuleList 是一个可以存放多个子模块的容器
        # 与普通 Python list 不同，它会被 PyTorch 正确识别和管理
        layers = []
        current_dims = input_dims

        for i, hidden_dim in enumerate(hidden_dims):
            # 全连接层: 将 current_dims 维的输入映射到 hidden_dim 维
            # 内部参数: 权重矩阵 W (hidden_dim x current_dims) + 偏置 b (hidden_dim,)
            layers.append(nn.Linear(current_dims, hidden_dim))

            # ReLU 激活函数: f(x) = max(0, x)
            # 作用: 引入非线性，让网络能学习复杂的模式
            # 没有激活函数的话，多层线性变换等价于一层线性变换
            layers.append(nn.ReLU())

            # Dropout: 训练时随机将 20% 的神经元输出置为 0
            # 作用: 防止过拟合（模型在训练集上表现好，但在测试集上差）
            # 注意: 测试/推理时 Dropout 会自动关闭
            layers.append(nn.Dropout(0.2))

            current_dims = hidden_dim

        # 输出层: 将最后一个隐藏层的输出映射到类别数
        layers.append(nn.Linear(current_dims, num_classes))

        # nn.Sequential 按顺序串联所有层，forward 时依次执行
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播: 定义数据在网络中的流动路径

        参数:
            x (Tensor): 输入图片，形状为 (batch_size, 1, 28, 28)

        返回:
            Tensor: 每个类别的得分，形状为 (batch_size, 10)
        """
        # view 将图片从 (batch_size, 1, 28, 28) 展平为 (batch_size, 784)
        # -1 表示自动计算该维度的大小
        x = x.view(x.size(0), -1)

        # 通过网络层依次计算
        return self.model(x)


# ============================================================================
#  模型 2: CNN（卷积神经网络 / Convolutional Neural Network）
# ============================================================================

class CNN(nn.Module):
    """
    卷积神经网络（LeNet 风格）

    网络结构:
        输入 (1, 28, 28)
        → Conv2d(1→32, 3x3)  → ReLU       # 提取低级特征（边缘、角点）
        → Conv2d(32→64, 3x3) → ReLU       # 提取高级特征（笔画、形状）
        → MaxPool2d(2x2)                   # 降低分辨率，减少计算量
        → Dropout(25%)                     # 防止过拟合
        → 展平 → FC(9216→128) → ReLU → Dropout(50%)
        → FC(128→10)                       # 输出分类结果

    数据形状变化（以 batch_size=64 为例）:
        (64, 1, 28, 28)   输入: 64张 28x28 灰度图
        → (64, 32, 26, 26) 第一层卷积后: 32个 26x26 特征图
        → (64, 64, 24, 24) 第二层卷积后: 64个 24x24 特征图
        → (64, 64, 12, 12) 最大池化后: 64个 12x12 特征图
        → (64, 9216)       展平后: 64 x (64*12*12) = 64 x 9216
        → (64, 128)        全连接层后
        → (64, 10)         输出层: 每个样本 10 个类别得分

    为什么 CNN 比 MLP 更适合图像?
        - MLP 把图片展平成向量，丢失了像素之间的空间关系
        - CNN 的卷积核在图片上滑动，能捕捉局部的空间模式
        - CNN 的参数共享机制大大减少了参数量
    """

    def __init__(self, num_classes=10):
        """
        参数:
            num_classes (int): 输出类别数，默认 10
        """
        super().__init__()

        # ---- 卷积层部分（特征提取器） ----

        # 第一层卷积: 输入 1 通道（灰度），输出 32 通道，卷积核大小 3x3
        # 卷积操作: 用一个 3x3 的小窗口在图片上滑动，提取局部特征
        # 输出尺寸: (28 - 3 + 1) = 26 → (batch, 32, 26, 26)
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 输入通道数（灰度图为 1，RGB 图为 3）
            out_channels=32,  # 输出通道数（即卷积核的个数，每个核提取一种特征）
            kernel_size=3     # 卷积核大小 3x3
        )

        # 第二层卷积: 输入 32 通道，输出 64 通道，卷积核 3x3
        # 输出尺寸: (26 - 3 + 1) = 24 → (batch, 64, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # 最大池化层: 窗口大小 2x2，步长 2
        # 作用: 在每个 2x2 区域中取最大值，把图片缩小一半
        # 输出尺寸: 24 / 2 = 12 → (batch, 64, 12, 12)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dropout 层: 防止过拟合
        self.dropout1 = nn.Dropout(0.25)  # 卷积层之后，丢弃 25%
        self.dropout2 = nn.Dropout(0.5)   # 全连接层之后，丢弃 50%

        # ---- 全连接层部分（分类器） ----

        # 全连接层: 将卷积提取的特征映射到 128 维
        # 输入: 64通道 x 12高 x 12宽 = 9216 维
        self.fc1 = nn.Linear(64 * 12 * 12, 128)

        # 输出层: 128 维 → 10 类
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 输入图片，形状为 (batch_size, 1, 28, 28)

        返回:
            Tensor: 每个类别的得分，形状为 (batch_size, 10)
        """
        # 卷积 → 激活 → 卷积 → 激活 → 池化 → Dropout
        x = F.relu(self.conv1(x))       # (batch, 1, 28, 28) → (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))       # (batch, 32, 26, 26) → (batch, 64, 24, 24)
        x = self.pool(x)                # (batch, 64, 24, 24) → (batch, 64, 12, 12)
        x = self.dropout1(x)

        # 展平: 将多维特征图变成一维向量
        x = x.view(x.size(0), -1)       # (batch, 64, 12, 12) → (batch, 9216)

        # 全连接 → 激活 → Dropout → 输出
        x = F.relu(self.fc1(x))         # (batch, 9216) → (batch, 128)
        x = self.dropout2(x)
        x = self.fc2(x)                 # (batch, 128) → (batch, 10)

        return x


# ============================================================================
#  辅助函数
# ============================================================================

def get_model(model_name='cnn', num_classes=10):
    """
    根据名称创建模型实例

    参数:
        model_name (str): 模型名称，'mlp' 或 'cnn'
        num_classes (int): 分类类别数

    返回:
        nn.Module: 模型实例
    """
    if model_name == 'mlp':
        return MLP(num_classes=num_classes)
    elif model_name == 'cnn':
        return CNN(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型名称: {model_name}，请选择 'mlp' 或 'cnn'")


def count_parameters(model):
    """
    统计模型的可训练参数总量

    参数:
        model (nn.Module): PyTorch 模型

    返回:
        int: 可训练参数总数

    使用示例:
        >>> model = CNN()
        >>> print(f"模型参数量: {count_parameters(model):,}")
        模型参数量: 600,394
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 直接运行本文件可以查看模型结构 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("模型 1: MLP（多层感知机）")
    print("=" * 60)
    mlp = MLP()
    print(mlp)
    print(f"可训练参数量: {count_parameters(mlp):,}\n")

    # 用随机数据测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)  # 1 张 28x28 灰度图
    output = mlp(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output.detach().numpy().round(3)}")

    print("\n" + "=" * 60)
    print("模型 2: CNN（卷积神经网络）")
    print("=" * 60)
    cnn = CNN()
    print(cnn)
    print(f"可训练参数量: {count_parameters(cnn):,}\n")

    output = cnn(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output.detach().numpy().round(3)}")
