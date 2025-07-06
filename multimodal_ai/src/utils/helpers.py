"""
辅助工具函数
"""

import torch
import numpy as np
import random
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import os


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_info(model: torch.nn.Module) -> Dict[str, int]:
    """获取详细的参数信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """保存JSON文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: Union[str, Path]) -> Any:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]):
    """保存Pickle文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """加载Pickle文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_directories(*paths: Union[str, Path]):
    """创建目录"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """计算文件哈希值"""
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_device(device: str = 'auto') -> torch.device:
    """获取设备"""
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    
    # 系统内存
    memory = psutil.virtual_memory()
    
    result = {
        'system_memory_total': memory.total / 1024**3,  # GB
        'system_memory_used': memory.used / 1024**3,    # GB
        'system_memory_percent': memory.percent
    }
    
    # GPU内存
    if torch.cuda.is_available():
        result.update({
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'gpu_memory_max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        })
    
    return result


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    """计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """获取经过的时间"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()


class ProgressBar:
    """简单的进度条"""
    
    def __init__(self, total: int, width: int = 50, desc: str = ""):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current += step
        self._print_progress()
    
    def _print_progress(self):
        """打印进度条"""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '-' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "N/A"
        
        print(f'\r{self.desc} |{bar}| {self.current}/{self.total} '
              f'({percent:.1%}) ETA: {eta_str}', end='', flush=True)
        
        if self.current >= self.total:
            print()  # 换行
    
    def finish(self):
        """完成进度条"""
        self.current = self.total
        self._print_progress()


def batch_process(items: List[Any], batch_size: int, process_func, 
                 desc: str = "Processing", show_progress: bool = True):
    """批量处理"""
    results = []
    
    if show_progress:
        progress = ProgressBar(len(items), desc=desc)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        if show_progress:
            progress.update(len(batch))
    
    if show_progress:
        progress.finish()
    
    return results


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    return a / b if b != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制数值范围"""
    return max(min_val, min(value, max_val))


def normalize_text(text: str) -> str:
    """标准化文本"""
    import re
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空格
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def ensure_list(item: Union[Any, List[Any]]) -> List[Any]:
    """确保返回列表"""
    if isinstance(item, list):
        return item
    else:
        return [item]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """反展平字典"""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


class ConfigMerger:
    """配置合并器"""
    
    @staticmethod
    def merge(base_config: Dict[str, Any], 
              override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger.merge(result[key], value)
            else:
                result[key] = value
        
        return result


def retry(max_attempts: int = 3, delay: float = 1.0, 
          exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
