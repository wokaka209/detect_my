'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-19 19:59:00
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-20 10:28:39
FilePath: \detect_my\evaluate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Author: wokaka209
"""
模型评估模块：生成混淆矩阵和计算评价指标，支持多融合结果
使用YOLOv5自带的验证脚本进行评估，避免兼容性问题
"""

# 必须在最开始应用兼容性补丁
import compat_patch

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLOV5_ROOT = ROOT / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

from config import EVALUATION, INFERENCE, FUSION_DATASETS, TRAINING


def evaluate_single_dataset(dataset_key, weights_path=None):
    """
    评估单个融合结果数据集 - 使用YOLOv5自带的val.py
    
    Args:
        dataset_key: 数据集键名（如'fusion1'）
        weights_path: 可选，指定权重文件路径。如果未提供，使用默认路径
    """
    if dataset_key not in FUSION_DATASETS:
        raise ValueError(f'Unknown dataset key: {dataset_key}')
    
    dataset_config = FUSION_DATASETS[dataset_key]
    exp_name = f'{dataset_key}_exp'
    
    # 如果指定了weights_path，使用指定路径；否则使用默认路径
    if weights_path is None:
        weights_path = Path(TRAINING['project']) / exp_name / 'weights' / 'best.pt'
    else:
        weights_path = Path(weights_path)
    
    save_dir = Path(EVALUATION['save_dir']) / dataset_key
    
    if not weights_path.exists():
        raise FileNotFoundError(f'Model not found: {weights_path}')
    
    print(f'\n=== Evaluating dataset: {dataset_config["name"]} ===')
    print(f'Model: {weights_path}')
    
    # 使用评估包装器（应用兼容性补丁）
    cmd = [
        sys.executable,
        'eval_wrapper.py',
        '--weights', str(weights_path),
        '--data', dataset_config['yaml_path'],
        '--batch-size', '1',
        '--imgsz', str(INFERENCE['img_size']),
        '--conf-thres', str(EVALUATION['conf_thres']),
        '--iou-thres', str(EVALUATION['iou_thres']),
        '--project', str(save_dir.parent),
        '--name', save_dir.name,
        '--device', '',
        '--save-txt',
        '--save-conf',
        '--verbose'
    ]
    
    print(f'Command: {" ".join(cmd)}')
    
    result = subprocess.run(cmd, cwd=ROOT)
    
    if result.returncode == 0:
        print(f'\n=== Evaluation completed for {dataset_config["name"]} ===')
        print(f'Results saved to: {save_dir}')
    else:
        print(f'\n=== Evaluation failed for {dataset_config["name"]} ===')
    
    return result.returncode


def evaluate_all_datasets():
    """评估所有融合结果数据集"""
    for dataset_key in FUSION_DATASETS:
        evaluate_single_dataset(dataset_key)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on dataset')
    parser.add_argument('dataset_key', help='Dataset key (e.g., WT_dataset)')
    parser.add_argument('--weights', help='Optional path to weights file')
    args = parser.parse_args()
    
    evaluate_single_dataset(args.dataset_key, args.weights)
