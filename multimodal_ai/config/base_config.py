"""
基础配置类
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml


class BaseConfig:
    """基础配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 项目根目录
        self.project_root = Path(__file__).parent.parent
        
        # 基础路径配置
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.logs_dir = self.project_root / "logs"
        self.cache_dir = self.project_root / "cache"
        
        # 创建必要目录
        self._create_directories()
        
        # 设备配置
        self.device = "cuda" if self._is_cuda_available() else "cpu"
        self.num_workers = min(4, os.cpu_count() or 1)
        
        # 日志配置
        self.log_level = "INFO"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # API配置
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.api_debug = False
        
        # 加载外部配置文件
        if config_path:
            self.load_config(config_path)
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _is_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def load_config(self, config_path: str):
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 更新配置
        self.update_config(config_data)
    
    def update_config(self, config_dict: Dict[str, Any]):
        """更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self, config_path: str):
        """保存配置到文件"""
        config_dict = self.to_dict()
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    def __repr__(self):
        return f"BaseConfig({self.to_dict()})"
