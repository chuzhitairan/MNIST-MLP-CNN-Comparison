"""
MNIST 手写数字识别 —— 训练脚本

这是项目的主入口文件，负责：
1. 解析命令行参数（模型类型、学习率、训练轮数等）
2. 加载 MNIST 数据集
3. 创建模型、损失函数、优化器
4. 执行训练循环（前向传播 → 计算损失 → 反向传播 → 更新参数）
5. 每个 epoch 在测试集上评估模型
6. 使用 TensorBoard 记录训练过程
7. 保存最佳模型

使用方法:
    # 使用 CNN 模型训练（推荐）
    python train.py --model cnn --epochs 10

    # 使用 MLP 模型训练
    python train.py --model mlp --epochs 10

    # 查看所有可用参数
    python train.py --help

    # 查看 TensorBoard 训练曲线
    tensorboard --logdir runs
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
import model as model_module
import utils


def parse_args():
    """
    解析命令行参数

    返回:
        argparse.Namespace: 包含所有参数的对象
    """
    parser = argparse.ArgumentParser(
        description='MNIST 手写数字识别训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 自动显示默认值
    )

    # ---- 模型相关 ----
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='模型类型: mlp（多层感知机）或 cnn（卷积神经网络）')

    # ---- 训练超参数 ----
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数（遍历整个训练集的次数）')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='每个批次的样本数量')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率（控制每次参数更新的步长大小）')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD 动量（加速收敛并减少震荡）')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减 / L2 正则化（防止过拟合）')
    parser.add_argument('--step-size', type=int, default=5,
                        help='学习率衰减间隔（每隔多少个 epoch 衰减一次）')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='学习率衰减因子（每次衰减后 lr = lr * gamma）')

    # ---- 其他设置 ----
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（保证实验可复现）')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集存储目录')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='TensorBoard 日志目录')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='数据加载子进程数')

    return parser.parse_args()


def train_one_epoch(model, device, train_loader, optimizer, epoch, writer):
    """
    训练一个 epoch（遍历整个训练集一次）

    深度学习训练的核心循环，每个批次执行以下步骤：
    1. 前向传播: 将数据输入模型，得到预测结果
    2. 计算损失: 比较预测结果和真实标签，算出误差
    3. 反向传播: 计算损失对每个参数的梯度（偏导数）
    4. 更新参数: 根据梯度调整模型参数，使损失减小

    参数:
        model: 神经网络模型
        device: 计算设备（CPU 或 GPU）
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前训练轮数
        writer: TensorBoard 写入器

    返回:
        float: 本轮的平均训练损失
    """
    # 设置模型为训练模式
    # 这会启用 Dropout 和 BatchNorm 的训练行为
    model.train()

    total_loss = 0.0     # 累计损失
    correct = 0           # 正确预测数
    total = 0             # 总样本数

    # tqdm 创建进度条，显示训练进度
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

    for batch_idx, (data, target) in enumerate(progress_bar):
        # ---- 步骤 0: 将数据移到计算设备（CPU 或 GPU） ----
        data, target = data.to(device), target.to(device)

        # ---- 步骤 1: 清零梯度 ----
        # PyTorch 默认会累积梯度，所以每次迭代前需要清零
        optimizer.zero_grad()

        # ---- 步骤 2: 前向传播 ----
        # 将数据输入模型，得到每个类别的得分（logits）
        output = model(data)

        # ---- 步骤 3: 计算损失 ----
        # 交叉熵损失（Cross Entropy Loss）：
        # - 内部先对 output 做 softmax，将得分转化为概率
        # - 然后计算预测概率分布和真实标签之间的差异
        # - 损失越小，说明模型预测越接近真实标签
        loss = F.cross_entropy(output, target)

        # ---- 步骤 4: 反向传播 ----
        # 自动计算损失对所有可训练参数的梯度
        # 这是深度学习的核心：通过链式法则（chain rule）逐层计算梯度
        loss.backward()

        # ---- 步骤 5: 更新参数 ----
        # 根据梯度更新模型参数: param = param - lr * gradient
        optimizer.step()

        # ---- 统计训练指标 ----
        total_loss += loss.item() * data.size(0)
        # output.argmax(dim=1) 取得分最高的类别作为预测结果
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    # 计算平均损失和准确率
    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    # 记录到 TensorBoard
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def evaluate(model, device, test_loader, epoch, writer):
    """
    在测试集上评估模型性能

    参数:
        model: 神经网络模型
        device: 计算设备
        test_loader: 测试数据加载器
        epoch: 当前轮数
        writer: TensorBoard 写入器

    返回:
        tuple: (平均损失, 准确率)
    """
    # 设置模型为评估模式
    # 这会关闭 Dropout，BatchNorm 使用全局统计值
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() 禁用梯度计算
    # 评估时不需要计算梯度，可以节省内存和加速计算
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    accuracy = 100. * correct / total

    # 记录到 TensorBoard
    writer.add_scalar('Test/Loss', avg_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, args, device, log_name_override=None):
    """
    抽取出的核心训练函数，可被其他脚本(如 compare.py)复用。
    """
    # 定义优化器和学习率调度器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 设置 TensorBoard
    log_name = log_name_override if log_name_override else f'{args.model}_lr{args.lr}_bs{args.batch_size}'
    writer = SummaryWriter(os.path.join(args.log_dir, log_name))

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, dummy_input)

    # 存储训练过程的指标
    metrics = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'best_acc': 0.0,
        'training_time': 0.0
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, epoch, writer)
        
        # 在测试集上评估
        test_loss, test_acc = evaluate(model, device, test_loader, epoch, writer)

        # 记录指标
        metrics['train_losses'].append(train_loss)
        metrics['train_accs'].append(train_acc)
        metrics['test_losses'].append(test_loss)
        metrics['test_accs'].append(test_acc)

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

        # 打印本轮结果
        print(f'Epoch {epoch:>2d}/{args.epochs} | '
              f'训练损失: {train_loss:.4f} | '
              f'测试损失: {test_loss:.4f} | '
              f'测试准确率: {test_acc:.2f}% | '
              f'学习率: {current_lr:.6f}')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, f'best_{args.model}.pth')
            utils.save_model(model, save_path)
            print(f'  ★ 新的最佳准确率! 模型已保存到 {save_path}')

    # 保存最终模型
    final_path = os.path.join(args.save_dir, f'final_{args.model}.pth')
    utils.save_model(model, final_path)

    elapsed = time.time() - start_time
    writer.close()
    
    metrics['best_acc'] = best_acc
    metrics['training_time'] = elapsed
    
    return metrics


def main():
    """主训练流程"""

    # ==================== 1. 解析参数 ====================
    args = parse_args()

    print("=" * 50)
    print("MNIST 手写数字识别 —— 训练开始")
    print("=" * 50)

    # ==================== 2. 设置设备 ====================
    # 自动检测是否有 GPU 可用
    # GPU（显卡）比 CPU 快很多倍，因为它有大量并行计算核心
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ==================== 3. 设置随机种子 ====================
    # 固定随机种子可以让实验结果可复现
    # 神经网络中有很多随机因素：权重初始化、数据打乱、Dropout 等
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ==================== 4. 加载数据 ====================
    print(f"\n正在加载 MNIST 数据集...")
    train_loader, test_loader = dataset.get_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)} 张图片, {len(train_loader)} 个批次")
    print(f"测试集: {len(test_loader.dataset)} 张图片, {len(test_loader)} 个批次")

    # ==================== 5. 创建模型 ====================
    print(f"\n创建模型: {args.model.upper()}")
    model = model_module.get_model(args.model)
    model = model.to(device)  # 将模型移到计算设备上
    param_count = model_module.count_parameters(model)
    print(f"模型参数量: {param_count:,}")

    # ==================== 6. 打印训练配置 ====================
    print(f"\n训练配置:")
    print(f"  模型:       {args.model.upper()}")
    print(f"  训练轮数:   {args.epochs}")
    print(f"  批次大小:   {args.batch_size}")
    print(f"  学习率:     {args.lr}")
    print(f"  动量:       {args.momentum}")
    print(f"  权重衰减:   {args.weight_decay}")
    print(f"  学习率衰减: 每 {args.step_size} 个 epoch × {args.gamma}")
    print(f"  设备:       {device}")
    print(f"  TensorBoard: tensorboard --logdir {args.log_dir}")
    print()

    # ==================== 7. 执行训练 ====================
    metrics = train_model(model, train_loader, test_loader, args, device)

    # ==================== 8. 训练结束 ====================
    print()
    print("=" * 50)
    print("训练完成!")
    print(f"  总耗时:       {metrics['training_time']:.1f} 秒")
    print(f"  最佳准确率:   {metrics['best_acc']:.2f}%")
    print(f"  最佳模型保存: {os.path.join(args.save_dir, f'best_{args.model}.pth')}")
    print(f"  最终模型保存: {os.path.join(args.save_dir, f'final_{args.model}.pth')}")
    print(f"\n查看训练曲线:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
