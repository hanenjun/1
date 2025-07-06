"""
数据预处理器
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config):
        self.config = config
        
        # 文本配置
        self.max_text_length = getattr(config, 'max_text_length', 50)

        # 音频配置
        self.audio_feature_dim = getattr(config, 'audio_feature_dim', 128)
        self.max_audio_length = getattr(config, 'max_audio_length', 200)

        # 视觉配置
        self.video_channels = getattr(config, 'video_channels', 3)
        self.video_frames = getattr(config, 'video_frames', 8)
        self.video_height = getattr(config, 'video_height', 64)
        self.video_width = getattr(config, 'video_width', 64)
        
    def preprocess_text(self, text: str, tokenizer) -> Dict[str, torch.Tensor]:
        """预处理文本"""
        # 编码文本
        tokens = tokenizer.encode(text, max_length=self.max_text_length)
        
        # 创建注意力掩码
        attention_mask = (tokens != tokenizer.pad_token_id).float()
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask
        }
    
    def preprocess_audio(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预处理音频特征"""
        # 确保音频特征维度正确
        if audio_features.dim() == 1:
            # 如果是1D，扩展为2D
            audio_features = audio_features.unsqueeze(-1).repeat(1, self.audio_feature_dim)
        elif audio_features.size(-1) != self.audio_feature_dim:
            # 如果特征维度不匹配，进行投影
            audio_features = F.linear(
                audio_features, 
                torch.randn(self.audio_feature_dim, audio_features.size(-1))
            )
        
        # 截断或填充到固定长度
        seq_len = audio_features.size(0)
        if seq_len > self.max_audio_length:
            # 截断
            audio_features = audio_features[:self.max_audio_length]
            attention_mask = torch.ones(self.max_audio_length)
        else:
            # 填充
            padding = torch.zeros(self.max_audio_length - seq_len, self.audio_feature_dim)
            audio_features = torch.cat([audio_features, padding], dim=0)
            attention_mask = torch.cat([
                torch.ones(seq_len),
                torch.zeros(self.max_audio_length - seq_len)
            ])
        
        return {
            'features': audio_features,
            'attention_mask': attention_mask
        }
    
    def preprocess_video(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预处理视频帧"""
        # 确保视频格式正确 [channels, frames, height, width]
        if video_frames.dim() == 4:
            if video_frames.shape[1] == self.video_channels:
                # [frames, channels, height, width] -> [channels, frames, height, width]
                video_frames = video_frames.permute(1, 0, 2, 3)
        
        # 调整帧数
        current_frames = video_frames.size(1)
        if current_frames != self.video_frames:
            if current_frames > self.video_frames:
                # 均匀采样
                indices = torch.linspace(0, current_frames - 1, self.video_frames).long()
                video_frames = video_frames[:, indices]
            else:
                # 重复最后一帧
                last_frame = video_frames[:, -1:].repeat(1, self.video_frames - current_frames, 1, 1)
                video_frames = torch.cat([video_frames, last_frame], dim=1)
        
        # 调整空间尺寸
        if video_frames.shape[-2:] != (self.video_height, self.video_width):
            video_frames = F.interpolate(
                video_frames.view(-1, *video_frames.shape[-2:]).unsqueeze(1),
                size=(self.video_height, self.video_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1).view(self.video_channels, self.video_frames, self.video_height, self.video_width)
        
        # 标准化到[-1, 1]
        video_frames = (video_frames - 0.5) / 0.5
        
        # 创建注意力掩码
        attention_mask = torch.ones(self.video_frames)
        
        return {
            'frames': video_frames,
            'attention_mask': attention_mask
        }
    
    def preprocess_image(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预处理单张图像"""
        # 确保图像格式正确 [channels, height, width]
        if image.dim() == 4 and image.size(0) == 1:
            image = image.squeeze(0)
        
        # 调整空间尺寸
        if image.shape[-2:] != (self.video_height, self.video_width):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.video_height, self.video_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 转换为视频格式（重复帧）
        video_frames = image.unsqueeze(1).repeat(1, self.video_frames, 1, 1)
        
        # 标准化到[-1, 1]
        video_frames = (video_frames - 0.5) / 0.5
        
        # 创建注意力掩码
        attention_mask = torch.ones(self.video_frames)
        
        return {
            'frames': video_frames,
            'attention_mask': attention_mask
        }
    
    def batch_preprocess(self, batch_data: Dict[str, List], tokenizer) -> Dict[str, torch.Tensor]:
        """批量预处理"""
        processed_batch = {}
        
        # 预处理文本
        if 'texts' in batch_data:
            text_tokens = []
            text_masks = []
            for text in batch_data['texts']:
                processed = self.preprocess_text(text, tokenizer)
                text_tokens.append(processed['tokens'])
                text_masks.append(processed['attention_mask'])
            
            processed_batch['text_tokens'] = torch.stack(text_tokens)
            processed_batch['text_masks'] = torch.stack(text_masks)
        
        # 预处理音频
        if 'audio_features' in batch_data:
            audio_features = []
            audio_masks = []
            for audio in batch_data['audio_features']:
                processed = self.preprocess_audio(audio)
                audio_features.append(processed['features'])
                audio_masks.append(processed['attention_mask'])
            
            processed_batch['audio_features'] = torch.stack(audio_features)
            processed_batch['audio_masks'] = torch.stack(audio_masks)
        
        # 预处理视频
        if 'video_frames' in batch_data:
            video_frames = []
            video_masks = []
            for video in batch_data['video_frames']:
                processed = self.preprocess_video(video)
                video_frames.append(processed['frames'])
                video_masks.append(processed['attention_mask'])
            
            processed_batch['video_frames'] = torch.stack(video_frames)
            processed_batch['video_masks'] = torch.stack(video_masks)
        
        # 预处理图像
        if 'images' in batch_data:
            image_frames = []
            image_masks = []
            for image in batch_data['images']:
                processed = self.preprocess_image(image)
                image_frames.append(processed['frames'])
                image_masks.append(processed['attention_mask'])
            
            processed_batch['image_frames'] = torch.stack(image_frames)
            processed_batch['image_masks'] = torch.stack(image_masks)
        
        return processed_batch
    
    def create_attention_mask(self, sequence_length: int, valid_length: int) -> torch.Tensor:
        """创建注意力掩码"""
        mask = torch.zeros(sequence_length)
        mask[:valid_length] = 1
        return mask
    
    def pad_sequence(self, sequences: List[torch.Tensor], 
                    max_length: Optional[int] = None,
                    padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """填充序列到相同长度"""
        if max_length is None:
            max_length = max(seq.size(0) for seq in sequences)
        
        padded_sequences = []
        attention_masks = []
        
        for seq in sequences:
            seq_len = seq.size(0)
            if seq_len >= max_length:
                # 截断
                padded_seq = seq[:max_length]
                mask = torch.ones(max_length)
            else:
                # 填充
                padding_shape = (max_length - seq_len,) + seq.shape[1:]
                padding = torch.full(padding_shape, padding_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding], dim=0)
                mask = torch.cat([
                    torch.ones(seq_len),
                    torch.zeros(max_length - seq_len)
                ])
            
            padded_sequences.append(padded_seq)
            attention_masks.append(mask)
        
        return torch.stack(padded_sequences), torch.stack(attention_masks)
    
    def normalize_features(self, features: torch.Tensor, 
                          method: str = 'standard') -> torch.Tensor:
        """特征标准化"""
        if method == 'standard':
            # 标准化 (mean=0, std=1)
            mean = features.mean(dim=-1, keepdim=True)
            std = features.std(dim=-1, keepdim=True)
            return (features - mean) / (std + 1e-8)
        elif method == 'minmax':
            # 最小-最大标准化 [0, 1]
            min_val = features.min(dim=-1, keepdim=True)[0]
            max_val = features.max(dim=-1, keepdim=True)[0]
            return (features - min_val) / (max_val - min_val + 1e-8)
        elif method == 'tanh':
            # tanh标准化 [-1, 1]
            return torch.tanh(features)
        else:
            return features
