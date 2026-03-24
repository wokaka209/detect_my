'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-19 19:58:10
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-20 12:23:52
FilePath: \detect_my\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Author: wokaka209
"""
模型训练模块：使用YOLOv5框架训练目标检测模型，支持多融合结果
"""

import os
import sys
import subprocess
from pathlib import Path
from config import TRAINING, FUSION_DATASETS


def train_single_dataset(dataset_key, resume=False):
    """
    训练单个融合结果数据集
    
    Args:
        dataset_key: 数据集键名（如'fusion1'）
        resume: 是否从上次中断的检查点恢复训练
    """
    if dataset_key not in FUSION_DATASETS:
        raise ValueError(f'Unknown dataset key: {dataset_key}')
    
    dataset_config = FUSION_DATASETS[dataset_key]
    exp_name = f'{dataset_key}_exp'
    
    # 基础命令
    cmd = [
        sys.executable,
        'train_wrapper.py',
        '--data', dataset_config['yaml_path'],
        '--epochs', str(TRAINING['epochs']),
        '--batch-size', str(TRAINING['batch_size']),
        '--imgsz', str(TRAINING['img_size']),
        '--project', TRAINING['project'],
        '--name', exp_name,
        '--workers', str(TRAINING['workers']),
        '--device', TRAINING['device']
    ]
    
    # 恢复训练或新训练
    if resume:
        # 从last.pt恢复训练
        last_weights = Path(TRAINING['project']) / exp_name / 'weights' / 'last.pt'
        if last_weights.exists():
            cmd.extend(['--resume', str(last_weights)])
            print(f'Resuming training from: {last_weights}')
        else:
            print(f'Warning: No checkpoint found at {last_weights}, starting new training...')
            cmd.extend(['--weights', TRAINING['weights']])
    else:
        cmd.extend(['--weights', TRAINING['weights']])
    
    print(f'\n=== Training dataset: {dataset_config["name"]} ===')
    print(f'Command: {" ".join(cmd)}')
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f'Training completed for {dataset_config["name"]}!')
        print(f'Model saved to: {TRAINING["project"]}/{exp_name}/weights/best.pt')
    else:
        print(f'Training failed for {dataset_config["name"]}!')
    
    return result.returncode


def train_all_datasets():
    """训练所有融合结果数据集"""
    for dataset_key in FUSION_DATASETS:
        train_single_dataset(dataset_key)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train model on dataset')
    parser.add_argument('dataset_key', help='Dataset key (e.g., WT_dataset)')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()
    
    train_single_dataset(args.dataset_key, args.resume)
