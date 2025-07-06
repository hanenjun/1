"""
视觉处理模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    """视觉编码器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.channels = config['channels']
        self.height = config['height']
        self.width = config['width']
        self.frames = config['frames']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 3D卷积特征提取器
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(self.channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((self.frames, 4, 4))
        )
        
        # 特征投影层
        self.feature_projection = nn.Linear(256 * 4 * 4, self.embed_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.frames, self.embed_dim)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(self.embed_dim, self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, video_frames, attention_mask=None):
        """
        前向传播
        
        Args:
            video_frames: [batch_size, channels, frames, height, width] 视频帧
            attention_mask: [batch_size, frames] 注意力掩码
            
        Returns:
            encoded_features: [batch_size, frames, hidden_dim] 编码后的特征
        """
        batch_size = video_frames.shape[0]
        
        # 3D卷积特征提取
        conv_features = self.conv3d_layers(video_frames)  # [batch_size, 256, frames, 4, 4]
        
        # 重塑为序列格式
        conv_features = conv_features.permute(0, 2, 1, 3, 4)  # [batch_size, frames, 256, 4, 4]
        conv_features = conv_features.reshape(batch_size, self.frames, -1)  # [batch_size, frames, 256*4*4]
        
        # 特征投影
        embeddings = self.feature_projection(conv_features)  # [batch_size, frames, embed_dim]
        
        # 添加位置编码
        embeddings = embeddings + self.pos_embedding
        embeddings = self.layer_norm(embeddings)
        
        # 创建padding mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, self.frames, device=video_frames.device)
        
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer编码
        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 输出投影
        output = self.output_projection(encoded)
        output = self.dropout_layer(output)
        
        return output
    
    def get_pooled_output(self, encoded_features, attention_mask=None):
        """
        获取池化后的输出
        
        Args:
            encoded_features: [batch_size, frames, hidden_dim]
            attention_mask: [batch_size, frames]
            
        Returns:
            pooled_output: [batch_size, hidden_dim]
        """
        if attention_mask is None:
            # 简单平均池化
            return encoded_features.mean(dim=1)
        else:
            # 加权平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded_features)
            sum_embeddings = torch.sum(encoded_features * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            return sum_embeddings / (sum_mask + 1e-9)


class ImageEncoder(nn.Module):
    """图像编码器（单帧处理）"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.channels = config['channels']
        self.height = config['height']
        self.width = config['width']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        
        # 2D卷积特征提取器
        self.conv2d_layers = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 特征投影层
        self.feature_projection = nn.Linear(256 * 4 * 4, self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.hidden_dim)
        
        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, images):
        """
        前向传播
        
        Args:
            images: [batch_size, channels, height, width] 图像
            
        Returns:
            encoded_features: [batch_size, hidden_dim] 编码后的特征
        """
        # 2D卷积特征提取
        conv_features = self.conv2d_layers(images)  # [batch_size, 256, 4, 4]
        
        # 展平
        conv_features = conv_features.view(conv_features.size(0), -1)  # [batch_size, 256*4*4]
        
        # 特征投影
        embeddings = self.feature_projection(conv_features)  # [batch_size, embed_dim]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 输出投影
        output = self.output_projection(embeddings)  # [batch_size, hidden_dim]
        
        return output


class VisionPreprocessor(nn.Module):
    """视觉预处理器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.channels = config['channels']
        self.height = config['height']
        self.width = config['width']
        self.frames = config['frames']
        
    def preprocess_video(self, video_frames):
        """
        预处理视频帧
        
        Args:
            video_frames: [batch_size, channels, frames, height, width] 原始视频帧
            
        Returns:
            processed_frames: [batch_size, channels, frames, height, width] 处理后的视频帧
        """
        # 标准化到[-1, 1]
        processed_frames = (video_frames - 0.5) / 0.5
        
        # 调整尺寸（如果需要）
        if video_frames.shape[-2:] != (self.height, self.width):
            processed_frames = F.interpolate(
                processed_frames.view(-1, self.channels, video_frames.shape[-2], video_frames.shape[-1]),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            )
            processed_frames = processed_frames.view(
                video_frames.shape[0], self.channels, self.frames, self.height, self.width
            )
        
        return processed_frames
    
    def preprocess_image(self, images):
        """
        预处理图像
        
        Args:
            images: [batch_size, channels, height, width] 原始图像
            
        Returns:
            processed_images: [batch_size, channels, height, width] 处理后的图像
        """
        # 标准化到[-1, 1]
        processed_images = (images - 0.5) / 0.5
        
        # 调整尺寸（如果需要）
        if images.shape[-2:] != (self.height, self.width):
            processed_images = F.interpolate(
                images,
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            )
        
        return processed_images
    
    def image_to_video(self, images):
        """
        将图像转换为视频格式（重复帧）
        
        Args:
            images: [batch_size, channels, height, width] 图像
            
        Returns:
            video_frames: [batch_size, channels, frames, height, width] 视频帧
        """
        batch_size, channels, height, width = images.shape
        
        # 重复图像帧
        video_frames = images.unsqueeze(2).repeat(1, 1, self.frames, 1, 1)
        
        return video_frames
