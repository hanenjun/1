#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理测试
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.tokenizer import SimpleTokenizer, ChineseTokenizer
from src.data.dataset import MultiModalDataset, DialogueDataGenerator
from src.data.preprocessor import DataPreprocessor
from config.model_config import ModelConfig


class TestTokenizers:
    """测试分词器"""
    
    def test_simple_tokenizer(self):
        """测试简单分词器"""
        vocab_size = 1000
        tokenizer = SimpleTokenizer(vocab_size)
        
        # 测试基本功能
        assert len(tokenizer) == vocab_size
        assert tokenizer.vocab_size == vocab_size
        
        # 测试编码解码
        text = "hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert all(0 <= token < vocab_size for token in tokens)
    
    def test_chinese_tokenizer(self):
        """测试中文分词器"""
        vocab_size = 1000
        tokenizer = ChineseTokenizer(vocab_size)
        
        # 测试中文文本
        chinese_text = "你好世界"
        tokens = tokenizer.encode(chinese_text)
        decoded = tokenizer.decode(tokens)
        
        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert all(0 <= token < vocab_size for token in tokens)
        
        # 测试英文文本
        english_text = "hello world"
        tokens = tokenizer.encode(english_text)
        decoded = tokenizer.decode(tokens)
        
        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
    
    def test_tokenizer_special_tokens(self):
        """测试特殊标记"""
        tokenizer = SimpleTokenizer(1000)
        
        # 测试特殊标记
        pad_token = tokenizer.pad_token
        unk_token = tokenizer.unk_token
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        
        assert pad_token in tokenizer.word_to_id
        assert unk_token in tokenizer.word_to_id
        assert bos_token in tokenizer.word_to_id
        assert eos_token in tokenizer.word_to_id


class TestDialogueDataGenerator:
    """测试对话数据生成器"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.generator = DialogueDataGenerator(self.config)
    
    def test_generate_text_dialogue(self):
        """测试文本对话生成"""
        dialogue = self.generator.generate_text_dialogue()
        
        assert 'user' in dialogue
        assert 'assistant' in dialogue
        assert isinstance(dialogue['user'], str)
        assert isinstance(dialogue['assistant'], str)
        assert len(dialogue['user']) > 0
        assert len(dialogue['assistant']) > 0
    
    def test_generate_multimodal_sample(self):
        """测试多模态样本生成"""
        sample = self.generator.generate_multimodal_sample()
        
        assert 'text' in sample
        assert 'audio_features' in sample
        assert 'video_frames' in sample
        
        # 检查数据类型和形状
        assert isinstance(sample['text'], str)
        assert isinstance(sample['audio_features'], torch.Tensor)
        assert isinstance(sample['video_frames'], torch.Tensor)
        
        # 检查形状
        assert sample['audio_features'].shape == (self.config.audio_seq_length, self.config.audio_feature_dim)
        assert sample['video_frames'].shape == (
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )
    
    def test_generate_batch(self):
        """测试批量生成"""
        batch_size = 4
        batch = self.generator.generate_batch(batch_size)
        
        assert len(batch) == batch_size
        for sample in batch:
            assert 'text' in sample
            assert 'audio_features' in sample
            assert 'video_frames' in sample


class TestMultiModalDataset:
    """测试多模态数据集"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.dataset = MultiModalDataset(self.config, num_samples=100)
    
    def test_dataset_length(self):
        """测试数据集长度"""
        assert len(self.dataset) == 100
    
    def test_dataset_getitem(self):
        """测试数据集索引"""
        sample = self.dataset[0]
        
        # 检查返回的键
        expected_keys = [
            'text_tokens', 'audio_features', 'video_frames',
            'text_mask', 'audio_mask', 'vision_mask', 'target_tokens'
        ]
        for key in expected_keys:
            assert key in sample
        
        # 检查数据类型
        assert isinstance(sample['text_tokens'], torch.Tensor)
        assert isinstance(sample['audio_features'], torch.Tensor)
        assert isinstance(sample['video_frames'], torch.Tensor)
        assert isinstance(sample['text_mask'], torch.Tensor)
        assert isinstance(sample['audio_mask'], torch.Tensor)
        assert isinstance(sample['vision_mask'], torch.Tensor)
        assert isinstance(sample['target_tokens'], torch.Tensor)
        
        # 检查形状
        assert sample['text_tokens'].shape == (self.config.max_seq_length,)
        assert sample['audio_features'].shape == (self.config.audio_seq_length, self.config.audio_feature_dim)
        assert sample['video_frames'].shape == (
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )
    
    def test_dataset_collate_fn(self):
        """测试数据集批处理函数"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            collate_fn=self.dataset.collate_fn
        )
        
        batch = next(iter(dataloader))
        
        # 检查批次形状
        assert batch['text_tokens'].shape == (4, self.config.max_seq_length)
        assert batch['audio_features'].shape == (4, self.config.audio_seq_length, self.config.audio_feature_dim)
        assert batch['video_frames'].shape == (
            4,
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )


class TestDataPreprocessor:
    """测试数据预处理器"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        texts = ["你好世界", "hello world", "测试文本"]
        
        result = self.preprocessor.preprocess_text(texts)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        
        # 检查形状
        batch_size = len(texts)
        assert result['input_ids'].shape == (batch_size, self.config.max_seq_length)
        assert result['attention_mask'].shape == (batch_size, self.config.max_seq_length)
        
        # 检查数据类型
        assert result['input_ids'].dtype == torch.long
        assert result['attention_mask'].dtype == torch.float
    
    def test_preprocess_audio(self):
        """测试音频预处理"""
        batch_size = 3
        audio_features = [
            torch.randn(50, self.config.audio_feature_dim),  # 短音频
            torch.randn(150, self.config.audio_feature_dim), # 长音频
            torch.randn(100, self.config.audio_feature_dim)  # 正常音频
        ]
        
        result = self.preprocessor.preprocess_audio(audio_features)
        
        assert 'audio_features' in result
        assert 'attention_mask' in result
        
        # 检查形状
        assert result['audio_features'].shape == (batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        assert result['attention_mask'].shape == (batch_size, self.config.audio_seq_length)
    
    def test_preprocess_video(self):
        """测试视频预处理"""
        batch_size = 2
        videos = [
            torch.randn(self.config.image_channels, 4, self.config.image_height, self.config.image_width),  # 短视频
            torch.randn(self.config.image_channels, 12, self.config.image_height, self.config.image_width)  # 长视频
        ]
        
        result = self.preprocessor.preprocess_video(videos)
        
        assert 'video_frames' in result
        assert 'attention_mask' in result
        
        # 检查形状
        assert result['video_frames'].shape == (
            batch_size,
            self.config.image_channels,
            self.config.video_frames,
            self.config.image_height,
            self.config.image_width
        )
        assert result['attention_mask'].shape == (batch_size, self.config.video_frames)
    
    def test_preprocess_image(self):
        """测试图像预处理"""
        batch_size = 3
        images = [
            torch.randn(self.config.image_channels, self.config.image_height, self.config.image_width)
            for _ in range(batch_size)
        ]
        
        result = self.preprocessor.preprocess_image(images)
        
        assert 'image_frames' in result
        
        # 检查形状
        assert result['image_frames'].shape == (
            batch_size,
            self.config.image_channels,
            self.config.image_height,
            self.config.image_width
        )
    
    def test_create_attention_mask(self):
        """测试注意力掩码创建"""
        # 测试不同长度的序列
        lengths = [5, 10, 3, 8]
        max_length = 12
        
        mask = self.preprocessor.create_attention_mask(lengths, max_length)
        
        # 检查形状
        assert mask.shape == (len(lengths), max_length)
        
        # 检查掩码正确性
        for i, length in enumerate(lengths):
            # 有效位置应该是1
            assert torch.all(mask[i, :length] == 1)
            # 填充位置应该是0
            if length < max_length:
                assert torch.all(mask[i, length:] == 0)
    
    def test_normalize_features(self):
        """测试特征标准化"""
        features = torch.randn(10, 128)
        
        normalized = self.preprocessor.normalize_features(features)
        
        # 检查形状不变
        assert normalized.shape == features.shape
        
        # 检查标准化效果（均值接近0，标准差接近1）
        mean = normalized.mean(dim=0)
        std = normalized.std(dim=0)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
