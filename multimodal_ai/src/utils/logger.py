"""
日志工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import time


def setup_logger(name: str = 'multimodal_ai',
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """设置日志器"""
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'multimodal_ai') -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class TimedLogger:
    """带时间记录的日志器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
    
    def start(self, message: str):
        """开始计时并记录"""
        self.start_time = time.time()
        self.logger.info(f"开始: {message}")
    
    def end(self, message: str):
        """结束计时并记录"""
        if self.start_time is None:
            self.logger.warning("未调用start()方法")
            return
        
        elapsed = time.time() - self.start_time
        self.logger.info(f"完成: {message} (耗时: {elapsed:.2f}秒)")
        self.start_time = None
    
    def step(self, message: str):
        """记录步骤"""
        if self.start_time is None:
            self.logger.info(f"步骤: {message}")
        else:
            elapsed = time.time() - self.start_time
            self.logger.info(f"步骤: {message} (已耗时: {elapsed:.2f}秒)")


class ProgressLogger:
    """进度日志器"""
    
    def __init__(self, logger: logging.Logger, total: int, log_interval: int = 10):
        self.logger = logger
        self.total = total
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()
    
    def update(self, step: int = 1, message: str = ""):
        """更新进度"""
        self.current += step
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            progress = self.current / self.total * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            
            log_msg = f"进度: {self.current}/{self.total} ({progress:.1f}%) - 已耗时: {elapsed:.1f}s{eta_str}"
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
    
    def finish(self, message: str = ""):
        """完成进度"""
        elapsed = time.time() - self.start_time
        log_msg = f"完成: {self.total}/{self.total} (100%) - 总耗时: {elapsed:.1f}s"
        if message:
            log_msg += f" - {message}"
        
        self.logger.info(log_msg)


class MultiLevelLogger:
    """多级别日志器"""
    
    def __init__(self, name: str = 'multimodal_ai'):
        self.logger = logging.getLogger(name)
    
    def debug(self, message: str, **kwargs):
        """调试信息"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """一般信息"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告信息"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误信息"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误"""
        self.logger.critical(message, **kwargs)
    
    def log_config(self, config: dict):
        """记录配置信息"""
        self.info("配置信息:")
        for key, value in config.items():
            self.info(f"  {key}: {value}")
    
    def log_model_info(self, model):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("模型信息:")
        self.info(f"  总参数量: {total_params:,}")
        self.info(f"  可训练参数: {trainable_params:,}")
        self.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    def log_training_step(self, epoch: int, step: int, loss: float, lr: float = None):
        """记录训练步骤"""
        msg = f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}"
        if lr is not None:
            msg += f", LR: {lr:.6f}"
        self.info(msg)
    
    def log_validation_results(self, epoch: int, metrics: dict):
        """记录验证结果"""
        self.info(f"Epoch {epoch} 验证结果:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {metric}: {value:.4f}")
            else:
                self.info(f"  {metric}: {value}")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """记录异常信息"""
        import traceback
        
        error_msg = f"异常发生"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {str(exception)}"
        
        self.error(error_msg)
        self.error("异常堆栈:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.error(f"  {line}")


# 全局日志器实例
_global_logger = None


def get_global_logger() -> MultiLevelLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = MultiLevelLogger()
    return _global_logger


def setup_global_logger(log_file: Optional[str] = None, 
                       level: int = logging.INFO,
                       console_output: bool = True):
    """设置全局日志器"""
    setup_logger(
        name='multimodal_ai',
        level=level,
        log_file=log_file,
        console_output=console_output
    )
    
    global _global_logger
    _global_logger = MultiLevelLogger('multimodal_ai')
