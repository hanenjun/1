#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_ai import MultiModalAI
from src.models.text_model import TextEncoder, TextDecoder
from src.models.audio_model import AudioEncoder, AudioPreprocessor
from src.models.vision_model import VisionEncoder, VisionPreprocessor, ImageEncoder
from src.models.fusion_model import MultiModalFusion
from config.model_config import ModelConfig


class TestModelConfig:
    """测试模型配置"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ModelConfig()
        assert config.vocab_size > 0
        assert config.d_model > 0
        assert config.num_heads > 0
        assert config.num_layers > 0
    
    def test_config_methods(self):
        """测试配置方法"""
        config = ModelConfig()
        
        text_config = config.get_text_config()
        assert 'vocab_size' in text_config
        assert 'd_model' in text_config
        
        audio_config = config.get_audio_config()
        assert 'audio_feature_dim' in audio_config
        
        vision_config = config.get_vision_config()
        assert 'image_channels' in vision_config
        
        fusion_config = config.get_fusion_config()
        assert 'd_model' in fusion_config


class TestTextModels:
    """测试文本模型"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.batch_size = 2
        self.seq_length = 10
    
    def test_text_encoder(self):
        """测试文本编码器"""
        encoder = TextEncoder(self.config)
        
        # 测试输入
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        # 前向传播
        output = encoder(input_ids, attention_mask)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.seq_length, self.config.d_model)
        assert output.shape == expected_shape
    
    def test_text_decoder(self):
        """测试文本解码器"""
        decoder = TextDecoder(self.config)
        
        # 测试输入
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        encoder_hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        # 前向传播
        output = decoder(input_ids, encoder_hidden_states, attention_mask)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.seq_length, self.config.vocab_size)
        assert output.shape == expected_shape


class TestAudioModels:
    """测试音频模型"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.batch_size = 2
    
    def test_audio_preprocessor(self):
        """测试音频预处理器"""
        preprocessor = AudioPreprocessor(self.config)
        
        # 测试输入
        audio_features = torch.randn(self.batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        
        # 前向传播
        output = preprocessor(audio_features)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.config.audio_seq_length, self.config.d_model)
        assert output.shape == expected_shape
    
    def test_audio_encoder(self):
        """测试音频编码器"""
        encoder = AudioEncoder(self.config)
        
        # 测试输入
        audio_features = torch.randn(self.batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        attention_mask = torch.ones(self.batch_size, self.config.audio_seq_length)
        
        # 前向传播
        output = encoder(audio_features, attention_mask)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.config.audio_seq_length, self.config.d_model)
        assert output.shape == expected_shape


class TestVisionModels:
    """测试视觉模型"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.batch_size = 2
    
    def test_vision_preprocessor(self):
        """测试视觉预处理器"""
        preprocessor = VisionPreprocessor(self.config)
        
        # 测试视频输入
        video_frames = torch.randn(
            self.batch_size, 
            self.config.image_channels, 
            self.config.video_frames,
            self.config.image_height, 
            self.config.image_width
        )
        
        # 前向传播
        output = preprocessor(video_frames)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.config.video_frames, self.config.d_model)
        assert output.shape == expected_shape
    
    def test_image_encoder(self):
        """测试图像编码器"""
        encoder = ImageEncoder(self.config)
        
        # 测试图像输入
        image_frames = torch.randn(
            self.batch_size,
            self.config.image_channels,
            self.config.image_height,
            self.config.image_width
        )
        
        # 前向传播
        output = encoder(image_frames)
        
        # 检查输出形状
        expected_shape = (self.batch_size, 1, self.config.d_model)
        assert output.shape == expected_shape
    
    def test_vision_encoder(self):
        """测试视觉编码器"""
        encoder = VisionEncoder(self.config)
        
        # 测试视频输入
        video_frames = torch.randn(
            self.batch_size,
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )
        attention_mask = torch.ones(self.batch_size, self.config.video_frames)
        
        # 前向传播
        output = encoder(video_frames, None, attention_mask)
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.config.video_frames, self.config.d_model)
        assert output.shape == expected_shape


class TestFusionModel:
    """测试融合模型"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.batch_size = 2
        self.seq_length = 10
    
    def test_multimodal_fusion(self):
        """测试多模态融合"""
        fusion = MultiModalFusion(self.config)
        
        # 测试输入
        text_features = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        audio_features = torch.randn(self.batch_size, self.config.audio_seq_length, self.config.d_model)
        vision_features = torch.randn(self.batch_size, self.config.video_frames, self.config.d_model)
        
        # 前向传播
        output = fusion(text_features, audio_features, vision_features)
        
        # 检查输出形状
        total_seq_length = self.seq_length + self.config.audio_seq_length + self.config.video_frames
        expected_shape = (self.batch_size, total_seq_length, self.config.d_model)
        assert output.shape == expected_shape


class TestMultiModalAI:
    """测试完整的多模态AI模型"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.model = MultiModalAI(self.config)
        self.batch_size = 2
        self.seq_length = 10
    
    def test_model_creation(self):
        """测试模型创建"""
        assert self.model is not None
        assert hasattr(self.model, 'text_encoder')
        assert hasattr(self.model, 'audio_encoder')
        assert hasattr(self.model, 'vision_encoder')
        assert hasattr(self.model, 'fusion')
        assert hasattr(self.model, 'text_decoder')
    
    def test_forward_pass(self):
        """测试前向传播"""
        # 准备输入
        text_tokens = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        audio_features = torch.randn(self.batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        video_frames = torch.randn(
            self.batch_size,
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )
        
        # 创建掩码
        text_mask = torch.ones(self.batch_size, self.seq_length)
        audio_mask = torch.ones(self.batch_size, self.config.audio_seq_length)
        vision_mask = torch.ones(self.batch_size, self.config.video_frames)
        
        # 前向传播
        output = self.model(
            text_tokens=text_tokens,
            audio_features=audio_features,
            video_frames=video_frames,
            image_frames=None,
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask
        )
        
        # 检查输出形状
        expected_shape = (self.batch_size, self.seq_length, self.config.vocab_size)
        assert output.shape == expected_shape
    
    def test_parameter_count(self):
        """测试参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
        
        # 检查参数数量在合理范围内（约12.6M）
        assert 10_000_000 <= total_params <= 20_000_000


if __name__ == '__main__':
    pytest.main([__file__])
