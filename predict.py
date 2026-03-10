"""
MNIST 手写数字识别 —— 预测/推理脚本

训练完模型后，使用本脚本进行预测和可视化：
1. 加载训练好的模型（.pth 文件）
2. 从测试集中随机取若干样本
3. 用模型预测每张图片的数字
4. 用 matplotlib 可视化预测结果

使用方法:
    # 使用 CNN 模型预测（默认）
    python predict.py

    # 使用 MLP 模型预测
    python predict.py --model mlp

    # 显示更多样本
    python predict.py --num-samples 25

    # 指定模型文件
    python predict.py --model-path checkpoints/best_cnn.pth
"""

import argparse
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import torch

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import dataset
import model as model_module
import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='MNIST 手写数字识别 —— 预测脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='模型类型')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型文件路径（不指定则自动查找 checkpoints/best_<model>.pth）')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='预测并显示的样本数量')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据集存储目录')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（不设置则每次随机选不同样本）')
    return parser.parse_args()


def predict(model, device, images):
    """
    使用模型对一批图片进行预测

    参数:
        model: 训练好的模型
        device: 计算设备
        images (Tensor): 图片张量，形状 (N, 1, 28, 28)

    返回:
        tuple: (预测类别列表, 置信度列表)

    说明:
        模型输出的是 10 个"得分"（logits），数值越大表示越可能是那个数字。
        通过 softmax 将得分转化为概率（所有概率之和为 1）。
        最高概率对应的类别就是模型的预测结果。
    """
    model.eval()  # 设置为评估模式

    with torch.no_grad():  # 推理时不需要计算梯度
        images = images.to(device)
        output = model(images)

        # softmax 将 logits 转化为概率分布
        # dim=1 表示在类别维度上做 softmax
        probabilities = torch.softmax(output, dim=1)

        # 取概率最大的类别和对应的概率值
        confidence, predicted = probabilities.max(dim=1)

    return predicted.cpu().tolist(), confidence.cpu().tolist()


def visualize_predictions(images, labels, predictions, confidences, save_path='predictions.png'):
    """
    可视化预测结果

    参数:
        images: 图片张量
        labels: 真实标签
        predictions: 模型预测
        confidences: 预测置信度
        save_path: 图片保存路径
    """
    num_samples = len(images)
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # 显示图片（去掉通道维度，转为 28x28）
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')

        # 判断预测是否正确
        is_correct = predictions[i] == labels[i]
        color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'

        # 设置标题: 预测/真实 (置信度)
        ax.set_title(
            f'{symbol} pred:{predictions[i]} true:{labels[i]}\n'
            f'conf:{confidences[i]:.1%}',
            color=color,
            fontsize=10
        )
        ax.axis('off')

    # 隐藏多余子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测结果图片已保存到: {save_path}")

    # 尝试显示
    try:
        if matplotlib.get_backend() != 'agg':
            plt.show()
    except Exception:
        pass
    finally:
        plt.close()


def main():
    args = parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ---- 设置设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 确定模型文件路径 ----
    if args.model_path is None:
        args.model_path = f'checkpoints/best_{args.model}.pth'

    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 '{args.model_path}'")
        print(f"请先运行训练脚本: python train.py --model {args.model}")
        return

    # ---- 加载模型 ----
    print(f"加载 {args.model.upper()} 模型: {args.model_path}")
    model = model_module.get_model(args.model)
    model = utils.load_model(model, args.model_path, device=device)
    model = model.to(device)

    # ---- 加载测试数据 ----
    _, test_loader = dataset.get_data_loaders(
        batch_size=args.num_samples,
        data_dir=args.data_dir,
        num_workers=0  # 推理时用单进程即可
    )

    # ---- 随机选取一个批次 ----
    # 跳过随机数量的批次，以获取不同的样本
    num_batches = len(test_loader)
    skip = random.randint(0, num_batches - 1)
    for i, (images, labels) in enumerate(test_loader):
        if i == skip:
            break

    images = images[:args.num_samples]
    labels = labels[:args.num_samples]

    # ---- 进行预测 ----
    predictions, confidences = predict(model, device, images)

    # ---- 统计结果 ----
    correct = sum(p == l for p, l in zip(predictions, labels.tolist()))
    total = len(predictions)

    print(f"\n预测结果: {correct}/{total} 正确 ({100 * correct / total:.1f}%)")
    for i in range(total):
        status = "✓" if predictions[i] == labels[i].item() else "✗"
        print(f"  样本 {i+1:>2d}: 预测={predictions[i]}, "
              f"真实={labels[i].item()}, "
              f"置信度={confidences[i]:.1%} {status}")

    # ---- 可视化 ----
    visualize_predictions(images, labels.tolist(), predictions, confidences)


if __name__ == '__main__':
    main()
