# 红外与可见光图像融合目标检测系统

基于YOLOv5框架的红外与可见光图像融合结果目标检测系统，支持多融合结果数据集。

## 项目结构

```
detect_my/
├── config.py          # 配置文件：集中管理所有关键参数
├── preprocess.py      # 数据预处理模块：划分训练集/验证集/测试集
├── train.py          # 模型训练模块：使用YOLOv5训练模型
├── evaluate.py       # 模型评估模块：生成混淆矩阵和计算评价指标
├── fps_eval.py       # FPS实时性评估模块（预留接口）
├── main.py           # 主程序入口：集成所有模块
├── yolov5/           # YOLOv5框架（已集成）
└── data/             # 数据目录
    ├── fusion1_images/ # 融合结果1图片
    ├── fusion2_images/ # 融合结果2图片
    ├── fusion3_images/ # 融合结果3图片
    └── fusion*_data/   # 处理后的数据集
```

## 功能特性

### 1. 多融合结果支持
- 支持多个融合结果数据集同时处理
- 每个数据集独立训练和评估
- 统一的配置管理

### 2. 数据预处理
- 按8:1:1比例划分训练集/验证集/测试集
- 自动创建YOLO格式的数据配置文件
- 支持随机种子确保可重复性

### 3. 模型训练
- 使用YOLOv5框架训练模型
- 可配置学习率、批次大小、训练轮数等超参数
- 自动保存模型权重和训练日志

### 4. 模型评估
- 生成混淆矩阵可视化
- 计算mAP@0.5、mAP@0.5:0.95
- 输出Precision、Recall、F1 Score等指标

### 5. FPS评估（预留接口）
- 支持实时性指标计算
- 可配置预热次数和测试次数
- 输出平均推理时间、FPS等指标

## 使用方法

### 准备数据

将融合结果图片放入对应的数据目录：
```
data/fusion1_images/  # 融合结果1
data/fusion2_images/  # 融合结果2
data/fusion3_images/  # 融合结果3
```

### 运行完整流程

```bash
# 处理所有融合结果数据集
python main.py --mode full

# 处理指定数据集
python main.py --mode full --dataset fusion1
```

### 分别运行各模块

```bash
# 数据预处理
python main.py --mode preprocess
python main.py --mode preprocess --dataset fusion1

# 模型训练
python main.py --mode train
python main.py --mode train --dataset fusion1

# 模型评估
python main.py --mode eval
python main.py --mode eval --dataset fusion1

# FPS评估（需在config.py中启用）
python main.py --mode fps
```

### 单独运行模块

```bash
# 数据预处理
python preprocess.py fusion1

# 模型训练
python train.py fusion1

# 模型评估
python evaluate.py fusion1
```

## 配置说明

所有关键参数集中在 [config.py](file:///e:/whx_Graduation%20project/baseline_project/detect_my/config.py) 中管理：

### 数据预处理参数
- `train_ratio`: 训练集比例（默认0.8）
- `val_ratio`: 验证集比例（默认0.1）
- `test_ratio`: 测试集比例（默认0.1）
- `random_seed`: 随机种子（默认42）

### 多融合结果数据集配置
在 `FUSION_DATASETS` 字典中配置每个数据集：
```python
FUSION_DATASETS = {
    'fusion1': {
        'name': 'fusion1',
        'source_dir': 'data/fusion1_images',
        'output_dir': 'data/fusion1_data',
        'yaml_path': 'data/fusion1_data.yaml',
        'nc': 1,
        'names': ['object']
    },
    # 添加更多数据集...
}
```

### 训练参数
- `weights`: 预训练权重（默认yolov5s.pt）
- `epochs`: 训练轮数（默认100）
- `batch_size`: 批次大小（默认16）
- `learning_rate`: 学习率（默认0.001）
- `img_size`: 输入图像尺寸（默认640）

### 推理参数
- `conf_thres`: 置信度阈值（默认0.25）
- `iou_thres`: IoU阈值（默认0.45）

### FPS评估配置
- `enable`: 是否启用FPS评估（默认False）
- `warmup_runs`: 预热次数（默认10）
- `test_runs`: 测试次数（默认100）

## 质量指标

- **混淆矩阵**：可视化分类性能
- **mAP@0.5**：平均精度（IoU=0.5）
- **mAP@0.5:0.95**：平均精度（IoU=0.5:0.05:0.95）
- **Precision**：精确率
- **Recall**：召回率
- **F1 Score**：精确率和召回率的调和平均
- **FPS**：每秒帧率（实时性指标）

## 输出结果

### 训练结果
- 模型权重：`runs/train/{dataset_name}_exp/weights/best.pt`
- 训练日志：`runs/train/{dataset_name}_exp/`
- 性能曲线：`runs/train/{dataset_name}_exp/results.png`

### 评估结果
- 混淆矩阵：`runs/eval/{dataset_name}/confusion_matrix.png`
- 评估报告：`runs/eval/{dataset_name}/evaluation_results.txt`

## 质量保证

- 代码遵循简洁高效原则
- 良好的可读性和可维护性
- 清晰的工程文件结构
- 关键步骤添加必要注释
- 重要参数配置集中管理

## 作者

wokaka209
