"""
文本处理模型
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.max_seq_length = config['max_seq_length']
        self.dropout = config['dropout']
        
        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoding = PositionalEncoding(self.embed_dim, self.max_seq_length)
        
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
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入token ids
            attention_mask: [batch_size, seq_len] 注意力掩码
            
        Returns:
            encoded_features: [batch_size, seq_len, hidden_dim] 编码后的特征
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # 创建padding mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # 转换为transformer需要的mask格式
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


class TextDecoder(nn.Module):
    """文本解码器"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.max_length = config['max_length']
        self.dropout = config['dropout']
        
        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoding = PositionalEncoding(self.hidden_dim, self.max_length)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, self.num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, target_ids, encoder_output, target_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            target_ids: [batch_size, tgt_len] 目标token ids
            encoder_output: [batch_size, src_len, hidden_dim] 编码器输出
            target_mask: [tgt_len, tgt_len] 目标序列掩码
            memory_mask: [batch_size, src_len] 记忆掩码
            
        Returns:
            logits: [batch_size, tgt_len, vocab_size] 输出logits
        """
        batch_size, tgt_len = target_ids.shape
        
        # 目标嵌入
        tgt_embeddings = self.embedding(target_ids)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        tgt_embeddings = self.pos_encoding(tgt_embeddings)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        
        # 创建因果掩码
        if target_mask is None:
            target_mask = self._generate_square_subsequent_mask(tgt_len).to(target_ids.device)
        
        # 创建记忆掩码
        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = (memory_mask == 0)
        
        # Transformer解码
        decoded = self.transformer(
            tgt_embeddings,
            encoder_output,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 输出投影
        logits = self.output_projection(decoded)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
