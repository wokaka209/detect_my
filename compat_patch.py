'''
Author: wokaka209 1325536985@qq.com
Date: 2026-03-20 09:32:34
LastEditors: wokaka209 1325536985@qq.com
LastEditTime: 2026-03-22 20:47:57
FilePath: \detect_my\compat_patch.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
"""
scipy/numpy 兼容性补丁
在导入任何依赖库之前应用此补丁
"""
import sys
import numpy as np

def apply_compatibility_fix():
    """
    应用 scipy/numpy/pandas 兼容性修复
    必须在导入 scipy/seaborn/pandas 之前调用
    """
    # 修复1: numpy 2.0 兼容性 - _NoValue 属性
    if not hasattr(np, '_NoValue'):
        try:
            np._NoValue = np._globals._NoValue
        except:
            np._NoValue = None

    # 修复2: 通过monkey patch修复scipy导入时空数组类型错误
    _old_numpy_array = np.array
    
    def _array_patch(*args, **kwargs):
        try:
            return _old_numpy_array(*args, **kwargs)
        except TypeError as e:
            # 捕获空数组类型错误
            if len(args) > 0 and args[0] == []:
                # 返回一个空数组带有dtype
                return _old_numpy_array([], dtype=np.float64)
            raise
    
    np.array = _array_patch
    
    # 修复3: pandas - numpy 版本兼容性 (TypeError: Cannot convert numpy.ndarray to numpy.ndarray)
    try:
        # Monkey patch pandas 的 maybe_convert_objects 函数
        import pandas
        from pandas.core.dtypes import cast
        from pandas.core import construction
        
        # 保存原始函数
        _old_maybe_convert_platform = cast.maybe_convert_platform
        _old_sanitize_array = construction.sanitize_array
        
        def _patched_maybe_convert_platform(data):
            """修复 numpy 数组转换问题"""
            if isinstance(data, np.ndarray):
                # 直接返回数组，避免重复转换
                return data
            try:
                return _old_maybe_convert_platform(data)
            except TypeError:
                # 如果转换失败，尝试直接返回
                return data
        
        def _patched_sanitize_array(data, *args, **kwargs):
            """修复 sanitize_array 中的数组转换问题"""
            try:
                return _old_sanitize_array(data, *args, **kwargs)
            except TypeError as e:
                if "Cannot convert numpy.ndarray" in str(e):
                    # 直接返回原始数据，避免转换
                    return data
                raise
        
        cast.maybe_convert_platform = _patched_maybe_convert_platform
        construction.sanitize_array = _patched_sanitize_array
        
        # 修复4: 提前导入并修复 pandas.lib
        try:
            from pandas._libs import lib
            # 尝试提前触发可能失败的转换
            if hasattr(lib, 'maybe_convert_objects'):
                _old_lib_maybe_convert = lib.maybe_convert_objects
                
                def _patched_lib_maybe_convert(arr, *args, **kwargs):
                    try:
                        return _old_lib_maybe_convert(arr, *args, **kwargs)
                    except TypeError:
                        return arr
                
                lib.maybe_convert_objects = _patched_lib_maybe_convert
        except:
            pass
    except:
        pass
    
    # 尝试导入scipy来应用修复
    try:
        import scipy.interpolate
    except:
        pass
    
    # 尝试导入pandas来触发修复
    try:
        import pandas
    except:
        pass
    
    # 修复5: 直接修复 yolov5 export_formats 函数避免 pandas 问题
    try:
        # 预先导入 yolov5.export 并替换 export_formats 函数
        if 'yolov5.export' in sys.modules:
            export_module = sys.modules['yolov5.export']
        else:
            # 尝试导入 export 模块
            try:
                import yolov5.export as export_module
            except ImportError:
                try:
                    import export as export_module
                except:
                    export_module = None
        
        if export_module:
            # 创建一个不使用 pandas 的替代版本
            def export_formats_no_pandas():
                """替代 export_formats，不使用 pandas"""
                return [
                    ["PyTorch", "-", ".pt", True, True],
                    ["TorchScript", "torchscript", ".torchscript", True, True],
                    ["ONNX", "onnx", ".onnx", True, True],
                    ["OpenVINO", "openvino", "_openvino_model", True, False],
                    ["TensorRT", "engine", ".engine", False, True],
                    ["CoreML", "coreml", ".mlpackage", True, False],
                    ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
                    ["TensorFlow GraphDef", "pb", ".pb", True, True],
                    ["TensorFlow Lite", "tflite", ".tflite", True, False],
                    ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
                    ["TensorFlow.js", "tfjs", "_web_model", False, False],
                    ["PaddlePaddle", "paddle", "_paddle_model", True, True],
                ]
            
            # 保存原始函数引用
            if hasattr(export_module, 'export_formats'):
                export_module._export_formats_original = export_module.export_formats
                
                def patched_export_formats():
                    """修复后的 export_formats，返回简单列表避免 pandas 问题"""
                    try:
                        return export_module._export_formats_original()
                    except:
                        # 失败时返回无 pandas 的版本
                        return export_formats_no_pandas()
                
                export_module.export_formats = patched_export_formats
    except Exception as e:
        print(f"export_formats 修复警告: {e}")
        pass
    
    # 恢复原始np.array（可选，因为其他地方可能也需要）
    # np.array = _old_numpy_array
    
    return True

# 自动应用修复
apply_compatibility_fix()
