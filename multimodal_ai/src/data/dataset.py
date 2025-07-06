"""
多模态数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from .tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class DialogueDataGenerator:
    """对话数据生成器"""
    
    def __init__(self, config=None):
        self.config = config
        # 预定义的对话模板
        self.dialogue_templates = [
            {
                'context': "你好",
                'response': "你好！很高兴见到你。有什么我可以帮助你的吗？"
            },
            {
                'context': "今天天气怎么样",
                'response': "今天天气很不错，阳光明媚，适合外出活动。"
            },
            {
                'context': "你能做什么",
                'response': "我可以和你聊天，回答问题，帮助你解决各种问题。"
            },
            {
                'context': "告诉我一个笑话",
                'response': "为什么程序员喜欢黑暗？因为光明会产生bug！"
            },
            {
                'context': "你叫什么名字",
                'response': "我是一个多模态AI助手，可以处理文本、音频和视频。"
            },
            {
                'context': "再见",
                'response': "再见！希望我们下次还能聊天。"
            },
            {
                'context': "谢谢你",
                'response': "不客气！很高兴能帮助到你。"
            },
            {
                'context': "你多大了",
                'response': "作为AI，我没有年龄概念，但我一直在学习和成长。"
            },
            {
                'context': "你喜欢什么",
                'response': "我喜欢学习新知识，帮助人们解决问题，和大家交流。"
            },
            {
                'context': "现在几点了",
                'response': "抱歉，我无法获取实时时间，建议你查看设备上的时钟。"
            },
            {
                'context': "你会说中文吗",
                'response': "是的，我可以用中文和你交流。"
            },
            {
                'context': "介绍一下自己",
                'response': "我是一个多模态AI助手，能够理解文本、音频和视觉信息，并与用户进行自然对话。"
            }
        ]
        
        # 扩展对话数据
        self.extended_dialogues = self._generate_extended_dialogues()
    
    def _generate_extended_dialogues(self):
        """生成扩展的对话数据"""
        extended = []
        
        # 基于模板生成变体
        variations = [
            ("你好", ["嗨", "您好", "hello", "hi", "早上好", "下午好"]),
            ("天气", ["气候", "温度", "weather", "气温"]),
            ("什么", ["啥", "哪些", "what", "什么样"]),
            ("怎么样", ["如何", "怎样", "how", "怎么办"]),
            ("谢谢", ["感谢", "thanks", "thank you", "多谢"]),
            ("再见", ["拜拜", "goodbye", "bye", "回见"])
        ]
        
        for template in self.dialogue_templates:
            extended.append(template)
            
            # 生成变体
            context = template['context']
            response = template['response']
            
            for original, variants in variations:
                if original in context:
                    for variant in variants:
                        new_context = context.replace(original, variant)
                        extended.append({
                            'context': new_context,
                            'response': response
                        })
        
        return extended
    
    def get_dialogue_pairs(self, num_samples: int = 1000) -> List[Dict[str, str]]:
        """获取对话对"""
        dialogues = []
        for _ in range(num_samples):
            dialogue = random.choice(self.extended_dialogues)
            dialogues.append(dialogue)
        return dialogues

    def generate_multimodal_sample(self) -> Dict[str, Any]:
        """生成多模态样本"""
        # 获取配置，如果没有则使用默认值
        if self.config:
            audio_seq_length = self.config.audio_seq_length
            audio_feature_dim = self.config.audio_feature_dim
            image_channels = self.config.image_channels
            video_frames = self.config.video_frames
            image_height = self.config.image_height
            image_width = self.config.image_width
        else:
            # 默认配置
            audio_seq_length = 100
            audio_feature_dim = 128
            image_channels = 3
            video_frames = 8
            image_height = 64
            image_width = 64

        # 生成文本
        dialogue = random.choice(self.extended_dialogues)
        text = dialogue['context'] + " " + dialogue['response']

        # 生成音频特征
        audio_features = torch.randn(audio_seq_length, audio_feature_dim)

        # 生成视频帧
        video_frames = torch.randn(image_channels, video_frames, image_height, image_width)

        return {
            'text': text,
            'audio_features': audio_features,
            'video_frames': video_frames
        }


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, 
                 num_samples: int = 5000,
                 vocab_size: int = 10000,
                 max_text_length: int = 50,
                 audio_feature_dim: int = 128,
                 video_frames: int = 8,
                 video_height: int = 64,
                 video_width: int = 64,
                 video_channels: int = 3):
        
        self.num_samples = num_samples
        self.max_text_length = max_text_length
        self.audio_feature_dim = audio_feature_dim
        self.video_frames = video_frames
        self.video_height = video_height
        self.video_width = video_width
        self.video_channels = video_channels
        
        # 初始化分词器
        self.tokenizer = SimpleTokenizer(vocab_size)
        
        # 生成对话数据
        self.dialogue_generator = DialogueDataGenerator()
        self.dialogues = self.dialogue_generator.get_dialogue_pairs(num_samples)
        
        # 构建词汇表
        all_texts = []
        for dialogue in self.dialogues:
            all_texts.append(dialogue['context'])
            all_texts.append(dialogue['response'])
        self.tokenizer.build_vocab(all_texts)
        
        logger.info(f"创建多模态数据集，样本数: {num_samples}")
        logger.info(f"词汇表大小: {len(self.tokenizer.word_to_id)}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx % len(self.dialogues)]
        
        # 文本处理
        context_tokens = self.tokenizer.encode(dialogue['context'], self.max_text_length)
        response_tokens = self.tokenizer.encode(dialogue['response'], self.max_text_length)
        
        # 生成合成音频特征 (模拟真实音频)
        audio_length = random.randint(50, 200)
        audio_features = torch.randn(audio_length, self.audio_feature_dim)
        
        # 生成合成视频特征 (模拟真实视频)
        video_frames = torch.randn(
            self.video_channels, self.video_frames, 
            self.video_height, self.video_width
        )
        
        # 创建注意力掩码
        context_mask = (context_tokens != self.tokenizer.pad_token_id).float()
        response_mask = (response_tokens != self.tokenizer.pad_token_id).float()
        audio_mask = torch.ones(audio_length)
        video_mask = torch.ones(self.video_frames)
        
        return {
            'context_tokens': context_tokens,
            'response_tokens': response_tokens,
            'audio_features': audio_features,
            'video_frames': video_frames,
            'context_mask': context_mask,
            'response_mask': response_mask,
            'audio_mask': audio_mask,
            'video_mask': video_mask,
            'context_text': dialogue['context'],
            'response_text': dialogue['response']
        }
    
    def get_tokenizer(self):
        """获取分词器"""
        return self.tokenizer


def collate_fn(batch):
    """批处理函数"""
    # 文本数据
    context_tokens = torch.stack([item['context_tokens'] for item in batch])
    response_tokens = torch.stack([item['response_tokens'] for item in batch])
    context_masks = torch.stack([item['context_mask'] for item in batch])
    response_masks = torch.stack([item['response_mask'] for item in batch])
    
    # 处理变长音频
    max_audio_len = max(item['audio_features'].size(0) for item in batch)
    audio_features = torch.zeros(len(batch), max_audio_len, batch[0]['audio_features'].size(1))
    audio_masks = torch.zeros(len(batch), max_audio_len)
    
    for i, item in enumerate(batch):
        audio_len = item['audio_features'].size(0)
        audio_features[i, :audio_len] = item['audio_features']
        audio_masks[i, :audio_len] = item['audio_mask'][:audio_len]
    
    # 视频数据
    video_frames = torch.stack([item['video_frames'] for item in batch])
    video_masks = torch.stack([item['video_mask'] for item in batch])
    
    # 文本数据
    context_texts = [item['context_text'] for item in batch]
    response_texts = [item['response_text'] for item in batch]
    
    return {
        'context_tokens': context_tokens,
        'response_tokens': response_tokens,
        'audio_features': audio_features,
        'video_frames': video_frames,
        'context_masks': context_masks,
        'response_masks': response_masks,
        'audio_masks': audio_masks,
        'video_masks': video_masks,
        'context_texts': context_texts,
        'response_texts': response_texts
    }


def create_dataloader(num_samples: int = 5000,
                     batch_size: int = 16,
                     vocab_size: int = 10000,
                     shuffle: bool = True,
                     num_workers: int = 0) -> Tuple[DataLoader, SimpleTokenizer]:
    """创建数据加载器"""
    dataset = MultiModalDataset(
        num_samples=num_samples,
        vocab_size=vocab_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return dataloader, dataset.get_tokenizer()


def create_train_val_dataloaders(num_train_samples: int = 4000,
                                num_val_samples: int = 1000,
                                batch_size: int = 16,
                                vocab_size: int = 10000,
                                num_workers: int = 0) -> Tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """创建训练和验证数据加载器"""
    # 创建训练集
    train_dataloader, tokenizer = create_dataloader(
        num_samples=num_train_samples,
        batch_size=batch_size,
        vocab_size=vocab_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 创建验证集（使用相同的分词器）
    val_dataset = MultiModalDataset(
        num_samples=num_val_samples,
        vocab_size=vocab_size
    )
    # 使用训练集的分词器
    val_dataset.tokenizer = tokenizer
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader, tokenizer
