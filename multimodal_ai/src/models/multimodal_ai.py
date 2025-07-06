"""
多模态AI主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List

from .text_model import TextEncoder, TextDecoder
from .audio_model import AudioEncoder, AudioPreprocessor
from .vision_model import VisionEncoder, VisionPreprocessor, ImageEncoder
from .fusion_model import MultiModalFusion


class MultiModalAI(nn.Module):
    """多模态AI主模型"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 获取各模态配置
        self.text_config = config.get_text_config()
        self.audio_config = config.get_audio_config()
        self.vision_config = config.get_vision_config()
        self.fusion_config = config.get_fusion_config()
        self.decoder_config = config.get_decoder_config()
        
        # 初始化各模态编码器
        self.text_encoder = TextEncoder(self.text_config)
        self.audio_encoder = AudioEncoder(self.audio_config)
        self.vision_encoder = VisionEncoder(self.vision_config)
        self.image_encoder = ImageEncoder(self.vision_config)
        
        # 初始化预处理器
        self.audio_preprocessor = AudioPreprocessor(self.audio_config)
        self.vision_preprocessor = VisionPreprocessor(self.vision_config)
        
        # 初始化融合模块
        self.fusion_module = MultiModalFusion(self.fusion_config)
        self.fusion = self.fusion_module  # 为了兼容性添加别名
        
        # 初始化解码器
        self.text_decoder = TextDecoder(self.decoder_config)
        
        # 特殊token的ID
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def forward(self, 
                text_tokens: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None,
                video_frames: Optional[torch.Tensor] = None,
                image_data: Optional[torch.Tensor] = None,
                target_tokens: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                audio_mask: Optional[torch.Tensor] = None,
                vision_mask: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            text_tokens: [batch_size, text_seq_len] 文本token
            audio_features: [batch_size, audio_seq_len, audio_dim] 音频特征
            video_frames: [batch_size, channels, frames, height, width] 视频帧
            image_data: [batch_size, channels, height, width] 图像数据
            target_tokens: [batch_size, target_seq_len] 目标token（训练时使用）
            text_mask: [batch_size, text_seq_len] 文本掩码
            audio_mask: [batch_size, audio_seq_len] 音频掩码
            vision_mask: [batch_size, vision_seq_len] 视觉掩码
            target_mask: [batch_size, target_seq_len] 目标掩码
            
        Returns:
            如果提供target_tokens，返回logits用于训练
            否则返回生成的token序列用于推理
        """
        # 编码各模态特征
        encoded_features = self._encode_modalities(
            text_tokens, audio_features, video_frames, image_data,
            text_mask, audio_mask, vision_mask
        )
        
        # 多模态融合
        fused_features = self.fusion_module(
            text_features=encoded_features.get('text'),
            audio_features=encoded_features.get('audio'),
            vision_features=encoded_features.get('vision'),
            text_mask=text_mask,
            audio_mask=audio_mask,
            vision_mask=vision_mask
        )
        
        # 解码生成文本
        if target_tokens is not None:
            # 训练模式：使用teacher forcing
            logits = self.text_decoder(
                target_tokens, fused_features, target_mask=None, memory_mask=None
            )
            return logits
        else:
            # 推理模式：自回归生成
            generated_tokens = self._generate_text(fused_features)
            return generated_tokens
    
    def _encode_modalities(self, text_tokens, audio_features, video_frames, image_data,
                          text_mask, audio_mask, vision_mask):
        """编码各模态特征"""
        encoded_features = {}
        
        # 文本编码
        if text_tokens is not None:
            encoded_features['text'] = self.text_encoder(text_tokens, text_mask)
        
        # 音频编码
        if audio_features is not None:
            # 预处理音频特征
            processed_audio = self.audio_preprocessor(audio_features)
            processed_audio, audio_mask = self.audio_preprocessor.pad_or_truncate(processed_audio)
            encoded_features['audio'] = self.audio_encoder(processed_audio, audio_mask)
        
        # 视觉编码
        if video_frames is not None:
            # 预处理视频帧
            processed_video = self.vision_preprocessor.preprocess_video(video_frames)
            encoded_features['vision'] = self.vision_encoder(processed_video, vision_mask)
        elif image_data is not None:
            # 处理单张图像
            processed_image = self.vision_preprocessor.preprocess_image(image_data)
            # 将图像转换为视频格式
            video_format = self.vision_preprocessor.image_to_video(processed_image)
            encoded_features['vision'] = self.vision_encoder(video_format, vision_mask)
        
        return encoded_features

    def _generate_text(self, encoder_output, max_length=None, temperature=1.0,
                      top_k=50, top_p=0.9):
        """
        自回归生成文本

        Args:
            encoder_output: [batch_size, seq_len, hidden_dim] 编码器输出
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k采样
            top_p: top-p采样

        Returns:
            generated_tokens: [batch_size, gen_len] 生成的token序列
        """
        if max_length is None:
            max_length = self.config.max_generate_length

        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # 初始化生成序列，以BOS token开始
        generated_tokens = torch.full(
            (batch_size, 1), self.bos_token_id,
            dtype=torch.long, device=device
        )

        # 自回归生成
        for _ in range(max_length - 1):
            # 获取当前序列的logits
            logits = self.text_decoder(
                generated_tokens, encoder_output, target_mask=None, memory_mask=None
            )

            # 取最后一个位置的logits
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k和Top-p采样
            next_token = self._sample_next_token(
                next_token_logits, top_k=top_k, top_p=top_p
            )

            # 添加到生成序列
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)

            # 检查是否所有序列都生成了EOS token
            if (next_token == self.eos_token_id).all():
                break

        return generated_tokens

    def _sample_next_token(self, logits, top_k=50, top_p=0.9):
        """
        采样下一个token

        Args:
            logits: [batch_size, vocab_size] 词汇表上的logits
            top_k: top-k采样参数
            top_p: top-p采样参数

        Returns:
            next_token: [batch_size] 下一个token
        """
        # Top-k采样
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

            # 将非top-k的logits设为负无穷
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered

        # Top-p采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 找到累积概率超过top_p的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 将要移除的位置设为负无穷
            sorted_logits[sorted_indices_to_remove] = float('-inf')

            # 恢复原始顺序
            logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))

        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        return next_token

    def chat(self, text_input: str, tokenizer, **kwargs) -> str:
        """
        文本对话接口

        Args:
            text_input: 输入文本
            tokenizer: 分词器
            **kwargs: 其他参数

        Returns:
            response: 响应文本
        """
        self.eval()
        with torch.no_grad():
            # 编码输入文本
            input_tokens = tokenizer.encode(text_input).unsqueeze(0)

            # 生成响应
            generated_tokens = self.forward(text_tokens=input_tokens)

            # 解码响应
            response = tokenizer.decode(generated_tokens[0])

        return response

    def chat_with_audio(self, audio_features: torch.Tensor,
                       text_context: str = None, tokenizer=None, **kwargs) -> str:
        """
        音频对话接口

        Args:
            audio_features: 音频特征
            text_context: 文本上下文
            tokenizer: 分词器
            **kwargs: 其他参数

        Returns:
            response: 响应文本
        """
        self.eval()
        with torch.no_grad():
            # 编码文本上下文（如果有）
            text_tokens = None
            if text_context and tokenizer:
                text_tokens = tokenizer.encode(text_context).unsqueeze(0)

            # 生成响应
            generated_tokens = self.forward(
                text_tokens=text_tokens,
                audio_features=audio_features.unsqueeze(0)
            )

            # 解码响应
            response = tokenizer.decode(generated_tokens[0]) if tokenizer else str(generated_tokens[0])

        return response

    def chat_with_vision(self, image_or_video: torch.Tensor,
                        text_context: str = None, tokenizer=None,
                        is_video: bool = False, **kwargs) -> str:
        """
        视觉对话接口

        Args:
            image_or_video: 图像或视频数据
            text_context: 文本上下文
            tokenizer: 分词器
            is_video: 是否为视频
            **kwargs: 其他参数

        Returns:
            response: 响应文本
        """
        self.eval()
        with torch.no_grad():
            # 编码文本上下文（如果有）
            text_tokens = None
            if text_context and tokenizer:
                text_tokens = tokenizer.encode(text_context).unsqueeze(0)

            # 生成响应
            if is_video:
                generated_tokens = self.forward(
                    text_tokens=text_tokens,
                    video_frames=image_or_video.unsqueeze(0)
                )
            else:
                generated_tokens = self.forward(
                    text_tokens=text_tokens,
                    image_data=image_or_video.unsqueeze(0)
                )

            # 解码响应
            response = tokenizer.decode(generated_tokens[0]) if tokenizer else str(generated_tokens[0])

        return response
