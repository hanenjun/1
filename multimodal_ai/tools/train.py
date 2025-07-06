#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_ai import MultiModalAI
from src.data.dataset import create_train_val_dataloaders
from config.training_config import TrainingConfig

# 尝试导入matplotlib，如果失败则跳过绘图功能
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib未安装，将跳过绘图功能")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = MultiModalAI(config)
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_dataloader, self.val_dataloader, self.tokenizer = create_train_val_dataloaders(
            num_train_samples=config.num_train_samples,
            num_val_samples=config.num_val_samples,
            batch_size=config.batch_size,
            vocab_size=config.vocab_size,
            num_workers=config.num_workers
        )
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 创建学习率调度器
        if config.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.min_learning_rate
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.no_improve_count = 0
        
        logger.info(f"训练器初始化完成")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {config.num_train_samples}")
        logger.info(f"验证样本数: {config.num_val_samples}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            context_tokens = batch['context_tokens'].to(self.device)
            response_tokens = batch['response_tokens'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_frames = batch['video_frames'].to(self.device)
            context_masks = batch['context_masks'].to(self.device)
            response_masks = batch['response_masks'].to(self.device)
            audio_masks = batch['audio_masks'].to(self.device)
            video_masks = batch['video_masks'].to(self.device)
            
            # 前向传播
            logits = self.model(
                text_tokens=context_tokens,
                audio_features=audio_features,
                video_frames=video_frames,
                target_tokens=response_tokens,
                text_mask=context_masks,
                audio_mask=audio_masks,
                vision_mask=video_masks,
                target_mask=response_masks
            )
            
            # 计算损失
            # 将logits和target重塑为2D
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = response_tokens.view(-1)
            
            loss = self.criterion(logits_flat, targets_flat)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动数据到设备
                context_tokens = batch['context_tokens'].to(self.device)
                response_tokens = batch['response_tokens'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_frames = batch['video_frames'].to(self.device)
                context_masks = batch['context_masks'].to(self.device)
                response_masks = batch['response_masks'].to(self.device)
                audio_masks = batch['audio_masks'].to(self.device)
                video_masks = batch['video_masks'].to(self.device)
                
                # 前向传播
                logits = self.model(
                    text_tokens=context_tokens,
                    audio_features=audio_features,
                    video_frames=video_frames,
                    target_tokens=response_tokens,
                    text_mask=context_masks,
                    audio_mask=audio_masks,
                    vision_mask=video_masks,
                    target_mask=response_masks
                )
                
                # 计算损失
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = response_tokens.view(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.to_dict(),
            'tokenizer_vocab': {
                'word_to_id': self.tokenizer.word_to_id,
                'id_to_word': self.tokenizer.id_to_word
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = self.config.checkpoints_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.config.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型到 {best_path}")
        
        logger.info(f"保存检查点到 {checkpoint_path}")
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            logger.info(f'Epoch {self.current_epoch}/{self.config.num_epochs}')
            logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
            
            # 保存检查点
            if self.current_epoch % self.config.save_interval == 0:
                self.save_checkpoint(is_best)
            
            # 早停
            if self.config.early_stopping and self.no_improve_count >= self.config.patience:
                logger.info(f"早停：验证损失连续{self.config.patience}个epoch没有改善")
                break
        
        # 保存最终模型
        self.save_checkpoint(is_best=True)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        logger.info("训练完成！")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib未安装，跳过绘制训练曲线")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plot_path = self.config.logs_dir / 'training_curves.png'
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"训练曲线保存到 {plot_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练多模态AI模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--num-epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainingConfig(args.config)
    else:
        config = TrainingConfig()
    
    # 覆盖命令行参数
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"从 {args.resume} 恢复训练")
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
