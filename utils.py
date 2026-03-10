"""
工具函数模块

提供训练过程中常用的辅助功能：
- 模型保存与加载
- 目录管理
"""

import os
from collections import OrderedDict

import torch


def ensure_dir(path):
    """
    确保目录存在，如果不存在则创建

    参数:
        path (str): 目录路径

    使用示例:
        >>> ensure_dir('./checkpoints')  # 自动创建 checkpoints 目录
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已创建目录: {path}")


def save_model(model, filepath):
    """
    保存模型的参数（state_dict）到文件

    说明:
        PyTorch 推荐只保存模型的参数（state_dict），而不是整个模型对象。
        这样更灵活、文件更小，加载时也不依赖具体的类定义路径。

    参数:
        model (nn.Module): 要保存的模型
        filepath (str): 保存路径，如 'checkpoints/best.pth'
    """
    # 确保保存目录存在
    dirpath = os.path.dirname(filepath)
    if dirpath:
        ensure_dir(dirpath)

    # 将所有参数移到 CPU 上保存（这样加载时不依赖 GPU）
    state_dict = OrderedDict()
    for key, value in model.state_dict().items():
        state_dict[key] = value.cpu()

    torch.save(state_dict, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(model, filepath, device='cpu'):
    """
    从文件加载模型参数

    参数:
        model (nn.Module): 模型实例（需要先创建好）
        filepath (str): 模型参数文件路径
        device (str): 加载到哪个设备，'cpu' 或 'cuda'

    返回:
        nn.Module: 加载了参数的模型

    使用示例:
        >>> model = CNN()
        >>> model = load_model(model, 'checkpoints/best.pth', device='cpu')
    """
    state_dict = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"已从 {filepath} 加载模型参数")
    return model


def evaluate_model(model, test_loader, device):
    """
    全方位评估模型性能，包括准确率、混淆矩阵、逐类别准确率和错误样本。
    """
    model.eval()
    all_preds = []
    all_targets = []
    wrong_samples = []  # 存储预测错误的样本：(image(cpu), pred, target)

    with torch.no_grad():
        for data, target in test_loader:
            data_device, target_device = data.to(device), target.to(device)
            output = model(data_device)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())

            # 收集错误样本（取最多一小部分为了可视化就够了）
            mismatch = pred != target_device
            if mismatch.any() and len(wrong_samples) < 20: # 限制数量以防内存过大
                mismatch_idxs = torch.where(mismatch)[0]
                for idx in mismatch_idxs:
                    if len(wrong_samples) < 20:
                        wrong_samples.append((
                            data[idx.item()].clone(), 
                            pred[idx.item()].item(), 
                            target[idx.item()].item()
                        ))

    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)

    # 总体准确率
    accuracy = 100. * (all_preds == all_targets).sum().item() / len(all_targets)

    # 混淆矩阵
    num_classes = 10
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        confusion_matrix[t.item(), p.item()] += 1

    # 逐类别准确率
    per_class_accuracy = []
    for i in range(num_classes):
        correct = confusion_matrix[i, i].item()
        total = confusion_matrix[i].sum().item()
        per_class_accuracy.append(100. * correct / total if total > 0 else 0.0)

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': confusion_matrix.numpy(),
        'wrong_samples': wrong_samples
    }


def measure_inference_time(model, test_loader, device, num_batches=20):
    """
    测量模型的平均推理时间（毫秒/单张图片）。
    """
    import time
    model.eval()
    
    # 预热 (warm-up)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    total_time = 0.0
    total_samples = 0
    batches_run = 0

    with torch.no_grad():
        for data, _ in test_loader:
            if batches_run >= num_batches:
                break
            data = data.to(device)
            
            # 使用 cuda 记录时间更准确（如果用 GPU）
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += data.size(0)
            batches_run += 1

    # 返回每张图片的平均推理时间（毫秒）
    return (total_time / total_samples) * 1000.0
