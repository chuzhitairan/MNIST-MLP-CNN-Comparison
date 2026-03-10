import argparse
import json
import os
import copy

import torch
import matplotlib.pyplot as plt
import numpy as np

import dataset
import model as model_module
import utils
from train import train_model


def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--num-workers', type=int, default=2)
    
    parser.add_argument('--experiments', type=str, default='all', choices=['default', 'sweep', 'all'])
    parser.add_argument('--output-dir', type=str, default='./results')
    
    args, _ = parser.parse_known_args()
    return args


def run_default_experiment(args, device, train_loader, test_loader):
    print("\n" + "="*50)
    print("运行默认对比实验: MLP vs CNN")
    print("="*50)
    
    results = {}
    
    for model_name in ['mlp', 'cnn']:
        print(f"\n[{model_name.upper()}] 开始训练和评估...")
        current_args = copy.copy(args)
        current_args.model = model_name
        
        torch.manual_seed(current_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_args.seed)
        
        model = model_module.get_model(model_name).to(device)
        param_count = model_module.count_parameters(model)
        
        train_metrics = train_model(
            model, train_loader, test_loader, current_args, device, 
            log_name_override=f'compare_{model_name}_default'
        )
        
        eval_metrics = utils.evaluate_model(model, test_loader, device)
        infer_time = utils.measure_inference_time(model, test_loader, device)
        
        results[model_name] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'inference_time_ms': infer_time,
            'param_count': param_count
        }
        
    return results


def run_hyperparameter_sweep(args, device, train_data_func):
    print("\n" + "="*50)
    print("运行超参数扫描实验")
    print("="*50)
    
    sweep_results = {'lr_sweep': {}, 'bs_sweep': {}}
    
    # ---------------- 学习率扫描 ----------------
    lr_list = [0.001, 0.01, 0.1]
    for lr in lr_list:
        sweep_results['lr_sweep'][lr] = {}
        for model_name in ['mlp', 'cnn']:
            print(f"\n[LR Sweep] 模型: {model_name.upper()}, LR: {lr}")
            current_args = copy.copy(args)
            current_args.model = model_name
            current_args.lr = lr
            current_args.epochs = 5  
            
            torch.manual_seed(current_args.seed)
            model = model_module.get_model(model_name).to(device)
            train_loader, test_loader = train_data_func(batch_size=current_args.batch_size)
            metrics = train_model(
                model, train_loader, test_loader, current_args, device,
                log_name_override=f'sweep_{model_name}_lr_{lr}'
            )
            sweep_results['lr_sweep'][lr][model_name] = metrics['best_acc']

    # ---------------- Batch Size 扫描 ----------------
    bs_list = [32, 64, 128, 256]
    for bs in bs_list:
        sweep_results['bs_sweep'][bs] = {}
        for model_name in ['mlp', 'cnn']:
            print(f"\n[Batch Size Sweep] 模型: {model_name.upper()}, BS: {bs}")
            current_args = copy.copy(args)
            current_args.model = model_name
            current_args.batch_size = bs
            current_args.epochs = 5
            
            torch.manual_seed(current_args.seed)
            model = model_module.get_model(model_name).to(device)
            train_loader, test_loader = train_data_func(batch_size=bs)
            metrics = train_model(
                model, train_loader, test_loader, current_args, device,
                log_name_override=f'sweep_{model_name}_bs_{bs}'
            )
            sweep_results['bs_sweep'][bs][model_name] = metrics['best_acc']
            
    return sweep_results


def generate_plots(default_res, sweep_res, output_dir):
    utils.ensure_dir(output_dir)
    
    plt.figure(figsize=(12, 10))
    metrics_to_plot = [
        ('Train Loss', 'train_losses'), ('Test Loss', 'test_losses'),
        ('Train Accuracy', 'train_accs'), ('Test Accuracy', 'test_accs')
    ]
    for i, (title, key) in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        if key in default_res['mlp']['train_metrics']:
            epochs = range(1, len(default_res['mlp']['train_metrics'][key]) + 1)
            plt.plot(epochs, default_res['mlp']['train_metrics'][key], 'o-', label='MLP')
            plt.plot(epochs, default_res['cnn']['train_metrics'][key], 's-', label='CNN')
            plt.title(title)
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.legend()
            plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, model_name in enumerate(['mlp', 'cnn']):
        cm = default_res[model_name]['eval_metrics']['confusion_matrix']
        cax = axes[i].matshow(cm, cmap='Blues')
        fig.colorbar(cax, ax=axes[i])
        axes[i].set_title(f'{model_name.upper()} Confusion Matrix', pad=20)
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        axes[i].set_xticks(range(10))
        axes[i].set_yticks(range(10))
        for x in range(10):
            for y in range(10):
                axes[i].text(x, y, str(cm[y, x]), va='center', ha='center',
                             color='white' if cm[y, x] > np.max(cm)/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(10)
    mlp_accs = default_res['mlp']['eval_metrics']['per_class_accuracy']
    cnn_accs = default_res['cnn']['eval_metrics']['per_class_accuracy']
    
    plt.bar(index, mlp_accs, bar_width, label='MLP')
    plt.bar(index + bar_width, cnn_accs, bar_width, label='CNN')
    plt.title('Per-Class Accuracy Comparison')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(index + bar_width / 2, range(10))
    plt.legend()
    plt.ylim(0, 105)
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    models = ['MLP', 'CNN']
    params = [default_res['mlp']['param_count'], default_res['cnn']['param_count']]
    times = [default_res['mlp']['inference_time_ms'], default_res['cnn']['inference_time_ms']]
    
    ax2 = ax1.twinx()
    ax1.bar(models, params, width=0.4, color='b', alpha=0.6, align='center', label='Parameters (Left)')
    ax2.plot(models, times, color='r', marker='o', linewidth=2, markersize=8, label='Inference Time (Right)')
    
    ax1.set_ylabel('Parameter Count', color='b')
    ax2.set_ylabel('Inference Time (ms / img)', color='r')
    plt.title('Model Complexity vs Inference Time')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_complexity.png'))
    plt.close()

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle('Sample Errors (Top: MLP, Bottom: CNN)')
    for i, model_name in enumerate(['mlp', 'cnn']):
        wrong_samples = default_res[model_name]['eval_metrics']['wrong_samples'][:8]
        for j, (img, pred, target) in enumerate(wrong_samples):
            ax = axes[i, j]
            img = img.squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"T:{target} P:{pred}", color='red')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_samples.png'))
    plt.close()

    if sweep_res:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        lrs = sorted(list(sweep_res['lr_sweep'].keys()))
        mlp_res_lr = [sweep_res['lr_sweep'][lr]['mlp'] for lr in lrs]
        cnn_res_lr = [sweep_res['lr_sweep'][lr]['cnn'] for lr in lrs]
        plt.plot(lrs, mlp_res_lr, 'o-', label='MLP')
        plt.plot(lrs, cnn_res_lr, 's-', label='CNN')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Learning Rate vs Accuracy (5 Epochs)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        bss = sorted(list(sweep_res['bs_sweep'].keys()))
        mlp_res_bs = [sweep_res['bs_sweep'][bs]['mlp'] for bs in bss]
        cnn_res_bs = [sweep_res['bs_sweep'][bs]['cnn'] for bs in bss]
        plt.plot(bss, mlp_res_bs, 'o-', label='MLP')
        plt.plot(bss, cnn_res_bs, 's-', label='CNN')
        plt.xlabel('Batch Size')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Batch Size vs Accuracy (5 Epochs)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hyperparameter_sweep.png'))
        plt.close()

def save_summary(default_res, sweep_res, output_dir):
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 40 + "\n")
        f.write("MNIST MLP vs CNN 对比实验总结\n")
        f.write("=" * 40 + "\n\n")
        
        for model in ['mlp', 'cnn']:
            f.write(f"【{model.upper()} 基本指标】\n")
            f.write(f"参数量: {default_res[model]['param_count']:,}\n")
            f.write(f"测试集准确率: {default_res[model]['train_metrics']['best_acc']:.2f}%\n")
            f.write(f"训练总耗时: {default_res[model]['train_metrics']['training_time']:.1f} 秒\n")
            f.write(f"单图平均推理时间: {default_res[model]['inference_time_ms']:.4f} ms\n\n")
        
        if sweep_res:
            f.write("【超参数扫描简报】\n")
            f.write("最佳学习率:\n")
            for model in ['mlp', 'cnn']:
                best_lr = max(sweep_res['lr_sweep'].keys(), key=lambda k: sweep_res['lr_sweep'][k][model])
                f.write(f"  {model.upper()}: {best_lr} (Acc: {sweep_res['lr_sweep'][best_lr][model]:.2f}%)\n")
            
            f.write("\n最佳 Batch Size:\n")
            for model in ['mlp', 'cnn']:
                best_bs = max(sweep_res['bs_sweep'].keys(), key=lambda k: sweep_res['bs_sweep'][k][model])
                f.write(f"  {model.upper()}: {best_bs} (Acc: {sweep_res['bs_sweep'][best_bs][model]:.2f}%)\n")

    json_data = {'default_experiment': {}, 'sweep_experiment': sweep_res}
    if default_res:
        json_data['default_experiment'] = default_res
        for m in ['mlp', 'cnn']:
            if 'eval_metrics' in json_data['default_experiment'][m]:
                if 'wrong_samples' in json_data['default_experiment'][m]['eval_metrics']:
                    del json_data['default_experiment'][m]['eval_metrics']['wrong_samples']
                if 'confusion_matrix' in json_data['default_experiment'][m]['eval_metrics']:
                    json_data['default_experiment'][m]['eval_metrics']['confusion_matrix'] = \
                        json_data['default_experiment'][m]['eval_metrics']['confusion_matrix'].tolist()

    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

def main():
    args = get_base_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    
    utils.ensure_dir(args.output_dir)

    def train_data_func(batch_size):
        return dataset.get_data_loaders(batch_size=batch_size, data_dir=args.data_dir, num_workers=args.num_workers)

    train_loader, test_loader = train_data_func(args.batch_size)

    default_res = run_default_experiment(args, device, train_loader, test_loader) if args.experiments in ['default', 'all'] else None
    sweep_res = run_hyperparameter_sweep(args, device, train_data_func) if args.experiments in ['sweep', 'all'] else None

    if default_res:
        generate_plots(default_res, sweep_res, args.output_dir)
        save_summary(default_res, sweep_res, args.output_dir)

if __name__ == '__main__':
    main()
