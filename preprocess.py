# Author: wokaka209
"""
数据预处理模块：将多个融合算法结果转换为YOLO格式数据集
支持配置多个独立的融合结果数据集，每个可包含一个或多个图像目录
"""

import os
import random
import shutil
from pathlib import Path
from config import DATA_SPLIT, FUSION_DATASETS


def collect_valid_images(image_dirs, label_dir):
    """
    从一个或多个图像目录收集所有有效图像（存在对应标签文件）
    
    Args:
        image_dirs: 图像目录列表（支持一个或多个目录）
        label_dir: 标签目录路径
    
    Returns:
        list: 有效图像文件的Path对象列表
    """
    label_path = Path(label_dir)
    valid_images = []
    
    for dir_path in image_dirs:
        img_path = Path(dir_path)
        if not img_path.exists():
            print(f"警告：图像目录不存在 - {dir_path}")
            continue
            
        # 收集所有图像文件
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_file in img_path.glob(ext):
                # 检查是否有对应标签文件
                label_file = label_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    valid_images.append(img_file)
    
    return valid_images


def split_dataset(image_dirs, label_dir, output_dir):
    """
    将数据集按比例划分为训练集、验证集和测试集
    支持从多个图像目录收集数据
    
    Args:
        image_dirs: 图像目录列表（支持一个或多个目录）
        label_dir: 标签目录路径
        output_dir: 输出目录路径
    """
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    
    random.seed(DATA_SPLIT['random_seed'])
    
    # 收集所有有效图像（支持多目录）
    image_files = collect_valid_images(image_dirs, label_dir)
    
    if not image_files:
        print(f"警告：在以下目录中未找到带对应标签的有效图像：")
        for d in image_dirs:
            print(f"  - {d}")
        return
    
    # 随机打乱
    random.shuffle(image_files)
    total = len(image_files)
    
    # 按比例划分
    train_count = int(total * DATA_SPLIT['train_ratio'])
    val_count = int(total * DATA_SPLIT['val_ratio'])
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    split_info = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # 创建输出目录并复制文件
    for split_name, files in split_info.items():
        # 创建目录
        img_output_dir = output_path / 'images' / split_name
        lbl_output_dir = output_path / 'labels' / split_name
        img_output_dir.mkdir(parents=True, exist_ok=True)
        lbl_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for img_file in files:
            # 复制图像
            dst_img = img_output_dir / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # 复制标签
            lbl_file = label_path / f"{img_file.stem}.txt"
            if lbl_file.exists():
                dst_lbl = lbl_output_dir / f"{img_file.stem}.txt"
                shutil.copy2(lbl_file, dst_lbl)
        
        print(f'  {split_name}: {len(files)} 张')
    
    print(f'\n  总计: {total} 张有效图像')
    print(f'  划分比例: 训练集={DATA_SPLIT["train_ratio"]}, 验证集={DATA_SPLIT["val_ratio"]}, 测试集={DATA_SPLIT["test_ratio"]}')


def create_data_yaml(dataset_config):
    """
    创建YOLO格式的数据配置文件
    
    Args:
        dataset_config: 数据集配置字典
    """
    output_path = Path(dataset_config['output_dir']).resolve()
    yaml_path = Path(dataset_config['yaml_path'])
    
    # 格式化类别名称
    names_str = ', '.join([f"'{name}'" for name in dataset_config['names']])
    
    yaml_content = f"""# YOLOv5 数据集配置
# 数据集名称: {dataset_config['name']}
# 图像来源目录: {', '.join(dataset_config['image_dirs'])}

path: {output_path.as_posix()}  # 数据集根目录（绝对路径）
train: images/train  # 训练集图像（相对path）
val: images/val      # 验证集图像（相对path）
test: images/test    # 测试集图像（相对path）

# 类别配置
nc: {dataset_config['nc']}  # 类别数量
names: [{names_str}]  # 类别名称
"""
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f'  配置文件: {yaml_path}')


def preprocess_single_dataset(dataset_key):
    """
    预处理单个融合结果数据集
    
    Args:
        dataset_key: 数据集键名（在FUSION_DATASETS中定义）
    """
    if dataset_key not in FUSION_DATASETS:
        available = list(FUSION_DATASETS.keys())
        raise ValueError(f'未知数据集: {dataset_key}\n可用数据集: {available}')
    
    dataset_config = FUSION_DATASETS[dataset_key]
    
    print(f'\n=== 处理数据集: {dataset_config["name"]} ===')
    print(f'  图像目录: {len(dataset_config["image_dirs"])} 个')
    
    split_dataset(
        dataset_config['image_dirs'],
        dataset_config['label_dir'],
        dataset_config['output_dir']
    )
    
    create_data_yaml(dataset_config)
    print(f'=== 完成: {dataset_config["name"]} ===\n')


def preprocess_all_datasets():
    """
    预处理所有配置的融合结果数据集
    """
    total = len(FUSION_DATASETS)
    print(f'=== 开始预处理所有 {total} 个数据集 ===')
    
    for i, dataset_key in enumerate(FUSION_DATASETS, 1):
        print(f'\n[{i}/{total}] ', end='')
        preprocess_single_dataset(dataset_key)
    
    print(f'=== 所有 {total} 个数据集处理完成 ===')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        dataset_key = sys.argv[1]
        if dataset_key.lower() == 'all':
            preprocess_all_datasets()
        else:
            preprocess_single_dataset(dataset_key)
    else:
        print("用法: python preprocess.py <数据集键名>")
        print(f"可用数据集: {list(FUSION_DATASETS.keys())}")
        print("\n示例:")
        print(f"  python preprocess.py {list(FUSION_DATASETS.keys())[0]}    # 处理第一个数据集")
        print(f"  python preprocess.py {list(FUSION_DATASETS.keys())[1]}    # 处理第二个数据集")
        print("  python preprocess.py all           # 处理所有数据集")
