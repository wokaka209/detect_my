'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-19 19:57:34
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-21 14:49:33
FilePath: \detect_my\config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Author: wokaka209
"""
配置文件：集中管理所有关键参数
支持多融合算法结果数据集配置
"""

# 数据预处理参数
DATA_SPLIT = {
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'test_ratio': 0.0,
    'random_seed': 42
}

# =============================================================================
# 多融合算法结果数据集配置
# 支持配置多个独立的融合算法结果数据集，每个有自己的图像和标签目录
# 每个数据集可以包含一个或多个图像目录（如一个算法产生的多个结果目录）
# =============================================================================

FUSION_DATASETS = {
    # 示例1：红外图像数据集（单个图像目录）
    'WT_dataset': {
        'name': 'WT_dataset',
        'image_dirs': [
            'E:/whx_Graduation project/baseline_project/wavelet_transform/results/WT_02_bad',
        ],
        'label_dir': 'E:/whx_Graduation project/baseline_project/dataset/labels',
        'output_dir': 'data/WT_yolo_dataset',
        'yaml_path': 'data/WT_data.yaml',
        'nc': 6,
        'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    },
    
    # 示例2：可见光图像数据集（单个图像目录）
    'LP_dataset': {
        'name': 'LP_dataset',
        'image_dirs': [
            'E:/whx_Graduation project/baseline_project/laplacian-pyramid-fusion/results/LPF_bad',
        ],
        'label_dir': 'E:/whx_Graduation project/baseline_project/dataset/labels',
        'output_dir': 'data/LP_yolo_dataset',
        'yaml_path': 'data/LP_data.yaml',
        'nc': 6,
        'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    },
    
    'mydesen_dataset': {
        'name': 'mydesen_dataset',
        'image_dirs': [
            'E:/whx_Graduation project/baseline_project/my_densefuse_advantive/data_result/batch_fusion_optimized',
        ],
        'label_dir': 'E:/whx_Graduation project/baseline_project/dataset/labels',
        'output_dir': 'data/mydesen_yolo_dataset',
        'yaml_path': 'data/mydesen_data.yaml',
        'nc': 6,
        'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    },
    
    'desen_dataset': {
        'name': 'desen_dataset',
        'image_dirs': [
            'E:/whx_Graduation project/baseline_project/DenseFuse_2019/data_result/batch_fusion_adaptive_l1',
        ],
        'label_dir': 'E:/whx_Graduation project/baseline_project/dataset/labels',
        'output_dir': 'data/desen_yolo_dataset',
        'yaml_path': 'data/desen_data.yaml',
        'nc': 6,
        'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    },
    
    # 示例3：融合算法1结果（单个图像目录）
    # 'fusion_result_1': {
    #     'name': 'fusion_algorithm_1',
    #     'image_dirs': [
    #         'E:/path/to/fusion1/images',
    #     ],
    #     'label_dir': 'E:/path/to/fusion1/labels',  # 可独立设置标签目录
    #     'output_dir': 'data/fusion1_dataset',
    #     'yaml_path': 'data/fusion1.yaml',
    #     'nc': 6,
    #     'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    # },
    
    # 示例4：融合算法2结果（多个图像目录，同一算法产生的多个结果）
    # 'fusion_result_2': {
    #     'name': 'fusion_algorithm_2',
    #     'image_dirs': [
    #         'E:/path/to/fusion2/result1',
    #         'E:/path/to/fusion2/result2',  # 支持同一算法的多个结果目录
    #     ],
    #     'label_dir': 'E:/path/to/fusion2/labels',
    #     'output_dir': 'data/fusion2_dataset',
    #     'yaml_path': 'data/fusion2.yaml',
    #     'nc': 6,
    #     'names': ['Bus', 'Car', 'Lamp', 'Motorcycle', 'People', 'Truck']
    # },
}

# 当前激活的默认数据集（用于main.py等脚本）
ACTIVE_DATASET = 'ir_dataset'

# 模型训练参数
TRAINING = {
    'weights': 'yolov5s.pt',
    'epochs': 100,
    'batch_size': 16,       # CPU训练减小批次大小
    'learning_rate': 0.001,
    'img_size': 640,
    'project': 'runs/train',
    'workers': 4,          # CPU训练使用0避免多进程问题
    'device': '',          # 自动选择设备（空字符串=自动检测，cuda:0=使用GPU，cpu=使用CPU）
    'multi_scale': False,  # 是否多尺度训练
    'save_period': 10,     # 每10轮保存一次权重
}

# 模型推理参数
INFERENCE = {
    'img_size': 640,
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'project': 'runs/detect',
    'save_txt': True,      # 保存检测结果为txt
    'save_conf': True,     # 保存置信度
}

# 评估参数
EVALUATION = {
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'save_dir': 'runs/eval',
    'save_json': False,    # 是否保存COCO格式的json结果
    'plot_confusion_matrix': True,  # 是否绘制混淆矩阵
}

# FPS评估参数（预留接口）
FPS_CONFIG = {
    'enable': False,
    'warmup_runs': 10,
    'test_runs': 100,
    'input_size': (3, 640, 640),  # 输入张量尺寸
}
