"""
训练配置类
"""

from .model_config import ModelConfig


class TrainingConfig(ModelConfig):
    """训练配置类"""
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        # 数据配置
        self.train_data_size = 2000
        self.val_data_size = 500
        self.test_data_size = 200
        self.data_split_ratio = [0.7, 0.2, 0.1]  # train, val, test
        self.num_train_samples = self.train_data_size
        self.num_val_samples = self.val_data_size
        
        # 训练超参数
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.min_learning_rate = 1e-6
        self.weight_decay = 1e-5
        self.num_epochs = 10
        self.warmup_steps = 100
        self.gradient_clip_norm = 1.0
        
        # 优化器配置
        self.optimizer = "adamw"  # adamw, adam, sgd
        self.scheduler = "cosine"  # cosine, linear, step
        self.use_scheduler = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        # 损失函数配置
        self.loss_function = "cross_entropy"
        self.label_smoothing = 0.1
        self.loss_weights = {
            'text': 1.0,
            'audio': 0.5,
            'vision': 0.5
        }
        
        # 验证和保存配置
        self.eval_steps = 100
        self.save_steps = 500
        self.logging_steps = 50
        self.log_interval = 10  # 每几个batch打印一次日志
        self.save_interval = 1  # 每几个epoch保存一次
        self.max_checkpoints = 5
        
        # 早停配置
        self.early_stopping = True
        self.patience = 3
        self.min_delta = 1e-4
        
        # 数据增强配置
        self.use_data_augmentation = True
        self.text_augmentation = {
            'synonym_replacement': 0.1,
            'random_insertion': 0.1,
            'random_swap': 0.1,
            'random_deletion': 0.1
        }
        
        # 音频增强配置
        self.audio_augmentation = {
            'noise_injection': 0.1,
            'time_stretch': 0.1,
            'pitch_shift': 0.1
        }
        
        # 视觉增强配置
        self.vision_augmentation = {
            'random_crop': 0.2,
            'random_flip': 0.5,
            'color_jitter': 0.1,
            'random_rotation': 15
        }
        
        # 分布式训练配置
        self.distributed = False
        self.local_rank = -1
        self.world_size = 1
        
        # 混合精度训练
        self.use_amp = True
        self.amp_opt_level = "O1"
        
        # 实验配置
        self.experiment_name = "multimodal_ai_experiment"
        self.run_name = None
        self.tags = ["multimodal", "transformer", "chat"]
        
        # 监控配置
        self.use_wandb = False
        self.use_tensorboard = True
        self.log_model_graph = True
        
    def get_optimizer_config(self):
        """获取优化器配置"""
        return {
            'name': self.optimizer,
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps
        }
    
    def get_scheduler_config(self):
        """获取学习率调度器配置"""
        return {
            'name': self.scheduler,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.num_epochs * (self.train_data_size // self.batch_size)
        }
    
    def get_data_config(self):
        """获取数据配置"""
        return {
            'train_size': self.train_data_size,
            'val_size': self.val_data_size,
            'test_size': self.test_data_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'split_ratio': self.data_split_ratio
        }
    
    def get_augmentation_config(self):
        """获取数据增强配置"""
        return {
            'text': self.text_augmentation,
            'audio': self.audio_augmentation,
            'vision': self.vision_augmentation
        } if self.use_data_augmentation else None
