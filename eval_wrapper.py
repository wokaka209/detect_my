'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-20 09:43:16
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-20 20:04:46
FilePath: \detect_my\eval_wrapper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# Author: wokaka209
"""
YOLOv5评估包装脚本：修复scipy/numpy兼容性问题
使用统一的兼容性补丁模块
"""
# 必须在最开始应用兼容性补丁 - 这是第一行可执行代码
import sys
import os

# 先添加当前目录到路径，确保能找到compat_patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 应用兼容性补丁
import compat_patch

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

# 运行原始验证脚本 - 使用与原始val.py相同的调用方式
if __name__ == '__main__':
    sys.path.insert(0, str(YOLOV5_ROOT))
    from val import main, parse_opt
    opt = parse_opt()
    main(opt)
