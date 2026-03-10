# MNIST 手写数字识别与模型对比

本项目为《人工智能导论》课程作业，基于 PyTorch 复现 MNIST 手写数字识别，并对比了多层感知机（MLP）与卷积神经网络（CNN）的性能差异。

## 项目结构

- `compare.py`：对比实验主脚本（生成图表与指标）
- `train.py` / `predict.py`：单模型训练与推理测试
- `model.py` / `dataset.py`：网络架构定义与数据加载
- `results/`：对比实验结果输出目录
- `report.md`：实验报告

## 环境配置

```bash
# 创建并激活虚拟环境 (Linux/Mac)
python -m venv venv
source venv/bin/activate  

# 安装依赖
pip install -r requirements.txt
```

## 运行实验

一键运行全量对比实验（含默认对比与超参数扫描），结果自动保存至 `results/` 目录：

```bash
./venv/bin/python compare.py --experiments all
```

**其他基础命令：**

```bash
# 单独训练测试 CNN
python train.py --model cnn --epochs 10

# 推理与可视化
python predict.py --model cnn --num-samples 16
```

## 致谢

参考开源项目：[pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)
