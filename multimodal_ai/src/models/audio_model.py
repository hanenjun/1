"""
音频处理模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.feature_dim = config['feature_dim']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.seq_length = config['seq_length']
        self.dropout = config['dropout']
        
        # 特征投影层
        self.feature_projection = nn.Linear(self.feature_dim, self.embed_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_length, self.embed_dim)
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
        
    def forward(self, audio_features, attention_mask=None):
        """
        前向传播
        
        Args:
            audio_features: [batch_size, seq_len, feature_dim] 音频特征
            attention_mask: [batch_size, seq_len] 注意力掩码
            
        Returns:
            encoded_features: [batch_size, seq_len, hidden_dim] 编码后的特征
        """
        batch_size, seq_len, _ = audio_features.shape
        
        # 特征投影
        embeddings = self.feature_projection(audio_features)
        
        # 添加位置编码
        if seq_len <= self.seq_length:
            pos_embeddings = self.pos_embedding[:, :seq_len, :]
        else:
            # 如果序列长度超过预设长度，进行插值
            pos_embeddings = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        embeddings = embeddings + pos_embeddings
        embeddings = self.layer_norm(embeddings)
        
        # 创建padding mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=audio_features.device)
        
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
            encoded_features: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
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


class AudioPreprocessor(nn.Module):
    """音频预处理器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.feature_dim = config['feature_dim']
        self.seq_length = config['seq_length']
        
        # 特征标准化
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # 卷积特征提取器（可选）
        self.use_conv = config.get('use_conv', False)
        if self.use_conv:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(self.feature_dim, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, self.feature_dim, kernel_size=3, padding=1)
            )
    
    def forward(self, audio_features):
        """
        预处理音频特征
        
        Args:
            audio_features: [batch_size, seq_len, feature_dim] 原始音频特征
            
        Returns:
            processed_features: [batch_size, seq_len, feature_dim] 处理后的特征
        """
        # 特征标准化
        features = self.feature_norm(audio_features)
        
        # 卷积处理（可选）
        if self.use_conv:
            # 转换维度用于卷积: [batch_size, feature_dim, seq_len]
            features_conv = features.transpose(1, 2)
            features_conv = self.conv_layers(features_conv)
            features = features_conv.transpose(1, 2)
        
        return features
    
    def pad_or_truncate(self, audio_features, target_length=None):
        """
        填充或截断音频特征到目标长度
        
        Args:
            audio_features: [batch_size, seq_len, feature_dim]
            target_length: 目标长度，默认使用配置中的seq_length
            
        Returns:
            processed_features: [batch_size, target_length, feature_dim]
            attention_mask: [batch_size, target_length]
        """
        if target_length is None:
            target_length = self.seq_length
        
        batch_size, seq_len, feature_dim = audio_features.shape
        
        if seq_len == target_length:
            attention_mask = torch.ones(batch_size, seq_len, device=audio_features.device)
            return audio_features, attention_mask
        elif seq_len < target_length:
            # 填充
            padding = torch.zeros(
                batch_size, target_length - seq_len, feature_dim,
                device=audio_features.device
            )
            padded_features = torch.cat([audio_features, padding], dim=1)
            
            # 创建注意力掩码
            attention_mask = torch.zeros(batch_size, target_length, device=audio_features.device)
            attention_mask[:, :seq_len] = 1
            
            return padded_features, attention_mask
        else:
            # 截断
            truncated_features = audio_features[:, :target_length, :]
            attention_mask = torch.ones(batch_size, target_length, device=audio_features.device)
            
            return truncated_features, attention_mask
