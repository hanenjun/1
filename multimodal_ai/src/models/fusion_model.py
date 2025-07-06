"""
多模态融合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.method = config['method']
        self.input_dims = config['input_dims']  # [text_dim, audio_dim, vision_dim]
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        
        # 输入投影层，将不同模态映射到统一维度
        self.text_projection = nn.Linear(self.input_dims[0], self.hidden_dim)
        self.audio_projection = nn.Linear(self.input_dims[1], self.hidden_dim)
        self.vision_projection = nn.Linear(self.input_dims[2], self.hidden_dim)
        
        # 根据融合方法选择不同的融合策略
        if self.method == 'attention':
            self._init_attention_fusion()
        elif self.method == 'concat':
            self._init_concat_fusion()
        elif self.method == 'add':
            self._init_add_fusion()
        else:
            raise ValueError(f"不支持的融合方法: {self.method}")
        
        # 输出投影层
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def _init_attention_fusion(self):
        """初始化注意力融合"""
        # 跨模态注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
    def _init_concat_fusion(self):
        """初始化拼接融合"""
        # 拼接后的维度是三个模态的和
        concat_dim = self.hidden_dim * 3
        self.concat_projection = nn.Sequential(
            nn.Linear(concat_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
    def _init_add_fusion(self):
        """初始化加法融合"""
        # 加法融合不需要额外参数，但可以添加权重
        self.text_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.vision_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, text_features=None, audio_features=None, vision_features=None,
                text_mask=None, audio_mask=None, vision_mask=None):
        """
        前向传播
        
        Args:
            text_features: [batch_size, text_seq_len, text_dim] 文本特征
            audio_features: [batch_size, audio_seq_len, audio_dim] 音频特征
            vision_features: [batch_size, vision_seq_len, vision_dim] 视觉特征
            text_mask: [batch_size, text_seq_len] 文本掩码
            audio_mask: [batch_size, audio_seq_len] 音频掩码
            vision_mask: [batch_size, vision_seq_len] 视觉掩码
            
        Returns:
            fused_features: [batch_size, seq_len, output_dim] 融合后的特征
        """
        # 收集有效的模态特征
        modality_features = []
        modality_masks = []
        
        if text_features is not None:
            text_proj = self.text_projection(text_features)
            modality_features.append(text_proj)
            modality_masks.append(text_mask)
            
        if audio_features is not None:
            audio_proj = self.audio_projection(audio_features)
            modality_features.append(audio_proj)
            modality_masks.append(audio_mask)
            
        if vision_features is not None:
            vision_proj = self.vision_projection(vision_features)
            modality_features.append(vision_proj)
            modality_masks.append(vision_mask)
        
        if not modality_features:
            raise ValueError("至少需要提供一种模态的特征")
        
        # 根据融合方法进行融合
        if self.method == 'attention':
            fused_features = self._attention_fusion(modality_features, modality_masks)
        elif self.method == 'concat':
            fused_features = self._concat_fusion(modality_features, modality_masks)
        elif self.method == 'add':
            fused_features = self._add_fusion(modality_features, modality_masks)
        
        # 输出投影
        output = self.output_projection(fused_features)
        output = self.dropout_layer(output)
        
        return output
    
    def _attention_fusion(self, modality_features, modality_masks):
        """注意力融合"""
        # 拼接所有模态特征
        all_features = torch.cat(modality_features, dim=1)  # [batch_size, total_seq_len, hidden_dim]

        # 创建统一的掩码
        all_masks = []
        for i, mask in enumerate(modality_masks):
            seq_len = modality_features[i].shape[1]
            batch_size = modality_features[i].shape[0]
            if mask is not None:
                # 确保掩码形状正确
                if mask.shape[1] != seq_len:
                    # 如果掩码长度不匹配，创建新的掩码
                    new_mask = torch.ones(batch_size, seq_len, device=modality_features[i].device, dtype=mask.dtype)
                    all_masks.append(new_mask)
                else:
                    all_masks.append(mask)
            else:
                # 创建全1掩码
                all_masks.append(torch.ones(batch_size, seq_len, device=modality_features[i].device))

        all_mask = torch.cat(all_masks, dim=1)
        key_padding_mask = (all_mask == 0)
        
        # 跨模态注意力
        attended_features, _ = self.cross_attention(
            all_features, all_features, all_features,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和层归一化
        attended_features = self.norm1(all_features + attended_features)
        
        # 前馈网络
        ffn_output = self.ffn(attended_features)
        fused_features = self.norm2(attended_features + ffn_output)
        
        return fused_features
    
    def _concat_fusion(self, modality_features, modality_masks):
        """拼接融合"""
        # 获取最大序列长度
        max_seq_len = max(feat.shape[1] for feat in modality_features)
        batch_size = modality_features[0].shape[0]
        
        # 填充所有特征到相同长度
        padded_features = []
        for feat in modality_features:
            if feat.shape[1] < max_seq_len:
                padding = torch.zeros(
                    batch_size, max_seq_len - feat.shape[1], self.hidden_dim,
                    device=feat.device
                )
                padded_feat = torch.cat([feat, padding], dim=1)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)
        
        # 在特征维度上拼接
        concat_features = torch.cat(padded_features, dim=-1)  # [batch_size, max_seq_len, hidden_dim*3]
        
        # 通过投影层降维
        fused_features = self.concat_projection(concat_features)
        
        return fused_features
    
    def _add_fusion(self, modality_features, modality_masks):
        """加法融合"""
        # 获取最大序列长度
        max_seq_len = max(feat.shape[1] for feat in modality_features)
        batch_size = modality_features[0].shape[0]
        
        # 初始化融合特征
        fused_features = torch.zeros(
            batch_size, max_seq_len, self.hidden_dim,
            device=modality_features[0].device
        )
        
        # 加权求和
        weights = [self.text_weight, self.audio_weight, self.vision_weight]
        for i, feat in enumerate(modality_features):
            if feat.shape[1] < max_seq_len:
                padding = torch.zeros(
                    batch_size, max_seq_len - feat.shape[1], self.hidden_dim,
                    device=feat.device
                )
                padded_feat = torch.cat([feat, padding], dim=1)
            else:
                padded_feat = feat
            
            fused_features += weights[i] * padded_feat
        
        return fused_features
    
    def get_pooled_output(self, fused_features, attention_mask=None):
        """
        获取池化后的输出
        
        Args:
            fused_features: [batch_size, seq_len, output_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, output_dim]
        """
        if attention_mask is None:
            # 简单平均池化
            return fused_features.mean(dim=1)
        else:
            # 加权平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(fused_features)
            sum_embeddings = torch.sum(fused_features * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            return sum_embeddings / (sum_mask + 1e-9)
