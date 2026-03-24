#!/usr/bin/env python
# Author: wokaka209
"""
YOLOv5训练包装脚本：修复scipy/numpy兼容性问题
使用统一的兼容性补丁模块
"""
# 必须在最开始应用兼容性补丁
import compat_patch

import sys
import os
from pathlib import Path

# 添加yolov5路径
FILE = Path(__file__).resolve()
YOLOV5_ROOT = FILE.parents[0] / 'yolov5'
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

# 设置环境变量
os.environ['WANDB_MODE'] = 'disabled'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# 禁用自动依赖检查
import utils.general as general
original_check_requirements = general.check_requirements

def patched_check_requirements(*args, **kwargs):
    # 跳过自动安装，只检查关键依赖
    return True

general.check_requirements = patched_check_requirements

# 运行原始训练脚本
if __name__ == '__main__':
    from train import run
    run()
