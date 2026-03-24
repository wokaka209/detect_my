# Author: wokaka209
"""
主程序入口：集成数据预处理、模型训练和评估模块，支持多融合结果
"""

# 必须在最开始应用兼容性补丁
import compat_patch

import sys
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
YOLOV5_ROOT = ROOT / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(YOLOV5_ROOT))

from preprocess import preprocess_single_dataset, preprocess_all_datasets
from train import train_single_dataset, train_all_datasets
from evaluate import evaluate_single_dataset, evaluate_all_datasets
from config import FUSION_DATASETS, FPS_CONFIG


class Pipeline:
    """完整的工作流管道"""
    
    def __init__(self):
        self.datasets = FUSION_DATASETS
        self.fps_config = FPS_CONFIG
    
    def run_preprocessing(self, dataset_key=None):
        """运行数据预处理"""
        print('=' * 50)
        print('Step 1: Data Preprocessing')
        print('=' * 50)
        
        if dataset_key:
            preprocess_single_dataset(dataset_key)
        else:
            preprocess_all_datasets()
        
        print('Data preprocessing completed.\n')
    
    def run_training(self, dataset_key=None, resume=False):
        """运行模型训练"""
        print('=' * 50)
        print('Step 2: Model Training')
        print('=' * 50)
        
        if dataset_key:
            train_single_dataset(dataset_key, resume)
        else:
            train_all_datasets()
        
        print('Model training completed.\n')
    
    def run_evaluation(self, dataset_key=None, weights_path=None):
        """运行模型评估"""
        print('=' * 50)
        print('Step 3: Model Evaluation')
        print('=' * 50)
        
        if dataset_key:
            evaluate_single_dataset(dataset_key, weights_path)
        else:
            evaluate_all_datasets()
        
        print('Model evaluation completed.\n')
    
    def run_fps_evaluation(self, dataset_key=None):
        """运行FPS实时性评估（预留接口）"""
        if not self.fps_config['enable']:
            print('FPS evaluation is disabled. Enable in config.py to use.\n')
            return
        
        print('=' * 50)
        print('Step 4: FPS Evaluation (Real-time Performance)')
        print('=' * 50)
        
        # TODO: 实现FPS计算逻辑
        print('FPS evaluation completed.\n')
    
    def run_full_pipeline(self, dataset_key=None):
        """运行完整的工作流"""
        self.run_preprocessing(dataset_key)
        self.run_training(dataset_key)
        self.run_evaluation(dataset_key)
        self.run_fps_evaluation(dataset_key)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Infrared and Visible Image Fusion Detection Pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'preprocess', 'train', 'eval', 'fps'],
                       help='运行模式: full(完整流程), preprocess(仅预处理), train(仅训练), eval(仅评估), fps(仅FPS评估)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='指定数据集键名（如fusion1），不指定则处理所有数据集')
    parser.add_argument('--weights', type=str, default=None,
                       help='指定权重文件路径（如runs/train/exp/weights/best.pt），仅在eval模式下有效')
    parser.add_argument('--resume', action='store_true',
                       help='恢复训练：从上次中断的检查点继续训练（仅在train模式下有效）')
    
    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    if args.dataset and args.dataset not in FUSION_DATASETS:
        available = ', '.join(FUSION_DATASETS.keys())
        print(f'Error: Unknown dataset "{args.dataset}"')
        print(f'Available datasets: {available}')
        sys.exit(1)
    
    if args.mode == 'full':
        pipeline.run_full_pipeline(args.dataset)
    elif args.mode == 'preprocess':
        pipeline.run_preprocessing(args.dataset)
    elif args.mode == 'train':
        pipeline.run_training(args.dataset, args.resume)
    elif args.mode == 'eval':
        pipeline.run_evaluation(args.dataset, args.weights)
    elif args.mode == 'fps':
        pipeline.run_fps_evaluation(args.dataset)


if __name__ == '__main__':
    main()
