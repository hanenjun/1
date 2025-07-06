"""
聊天API接口
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import base64
from io import BytesIO

from ..models.multimodal_ai import MultiModalAI
from ..data.tokenizer import SimpleTokenizer
from ..data.preprocessor import DataPreprocessor
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ChatAPI:
    """多模态聊天API"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: str = 'auto'):
        
        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载配置
        if config_path:
            self.config = ModelConfig()
            self.config.load_config(config_path)
        else:
            self.config = ModelConfig()
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(self.config.to_dict())
        
        # 初始化分词器
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        
        # 初始化模型
        self.model = MultiModalAI(self.config)
        self.model.to(self.device)
        
        # 加载模型权重
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # 对话历史
        self.conversation_history = []
        self.max_history_length = 10
        
        logger.info(f"ChatAPI初始化完成，设备: {self.device}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # 加载分词器
            if 'tokenizer_vocab' in checkpoint:
                tokenizer_data = checkpoint['tokenizer_vocab']
                self.tokenizer.word_to_id = tokenizer_data['word_to_id']
                self.tokenizer.id_to_word = {int(k): v for k, v in tokenizer_data['id_to_word'].items()}
            
            self.model.eval()
            logger.info(f"模型已从 {model_path} 加载")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def chat_text(self, 
                  text_input: str,
                  max_length: int = 50,
                  temperature: float = 1.0,
                  top_k: int = 50,
                  top_p: float = 0.9) -> Dict[str, Any]:
        """文本对话"""
        try:
            # 预处理输入
            processed_input = self.preprocessor.preprocess_text(text_input, self.tokenizer)
            text_tokens = processed_input['tokens'].unsqueeze(0).to(self.device)
            text_mask = processed_input['attention_mask'].unsqueeze(0).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                generated_tokens = self.model._generate_text(
                    encoder_output=self.model._encode_modalities(
                        text_tokens, None, None, None, text_mask, None, None
                    )['text'],
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # 解码响应
            response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # 更新对话历史
            self._add_to_history(text_input, response_text)
            
            return {
                'success': True,
                'response': response_text,
                'input': text_input,
                'metadata': {
                    'model_type': 'text_only',
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                }
            }
            
        except Exception as e:
            logger.error(f"文本对话失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "抱歉，处理您的请求时出现了错误。"
            }
    
    def chat_audio(self,
                   audio_features: Union[torch.Tensor, np.ndarray],
                   text_context: Optional[str] = None,
                   max_length: int = 50,
                   temperature: float = 1.0) -> Dict[str, Any]:
        """音频对话"""
        try:
            # 转换音频特征格式
            if isinstance(audio_features, np.ndarray):
                audio_features = torch.from_numpy(audio_features).float()
            
            # 预处理音频
            processed_audio = self.preprocessor.preprocess_audio(audio_features)
            audio_tensor = processed_audio['features'].unsqueeze(0).to(self.device)
            audio_mask = processed_audio['attention_mask'].unsqueeze(0).to(self.device)
            
            # 预处理文本上下文（如果有）
            text_tokens = None
            text_mask = None
            if text_context:
                processed_text = self.preprocessor.preprocess_text(text_context, self.tokenizer)
                text_tokens = processed_text['tokens'].unsqueeze(0).to(self.device)
                text_mask = processed_text['attention_mask'].unsqueeze(0).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                encoded_features = self.model._encode_modalities(
                    text_tokens, audio_tensor, None, None,
                    text_mask, audio_mask, None
                )
                
                # 融合特征
                fused_features = self.model.fusion_module(
                    text_features=encoded_features.get('text'),
                    audio_features=encoded_features.get('audio'),
                    vision_features=None,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    vision_mask=None
                )
                
                generated_tokens = self.model._generate_text(
                    fused_features, max_length=max_length, temperature=temperature
                )
            
            # 解码响应
            response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # 更新对话历史
            input_desc = f"[音频输入]" + (f" + {text_context}" if text_context else "")
            self._add_to_history(input_desc, response_text)
            
            return {
                'success': True,
                'response': response_text,
                'input_type': 'audio',
                'text_context': text_context,
                'metadata': {
                    'model_type': 'multimodal',
                    'audio_length': audio_features.shape[0],
                    'temperature': temperature
                }
            }
            
        except Exception as e:
            logger.error(f"音频对话失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "抱歉，处理音频输入时出现了错误。"
            }
    
    def chat_vision(self,
                    image_or_video: Union[torch.Tensor, np.ndarray],
                    text_context: Optional[str] = None,
                    is_video: bool = False,
                    max_length: int = 50,
                    temperature: float = 1.0) -> Dict[str, Any]:
        """视觉对话"""
        try:
            # 转换视觉数据格式
            if isinstance(image_or_video, np.ndarray):
                image_or_video = torch.from_numpy(image_or_video).float()
            
            # 预处理视觉数据
            if is_video:
                processed_vision = self.preprocessor.preprocess_video(image_or_video)
            else:
                processed_vision = self.preprocessor.preprocess_image(image_or_video)
            
            vision_tensor = processed_vision['frames'].unsqueeze(0).to(self.device)
            vision_mask = processed_vision['attention_mask'].unsqueeze(0).to(self.device)
            
            # 预处理文本上下文（如果有）
            text_tokens = None
            text_mask = None
            if text_context:
                processed_text = self.preprocessor.preprocess_text(text_context, self.tokenizer)
                text_tokens = processed_text['tokens'].unsqueeze(0).to(self.device)
                text_mask = processed_text['attention_mask'].unsqueeze(0).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                if is_video:
                    encoded_features = self.model._encode_modalities(
                        text_tokens, None, vision_tensor, None,
                        text_mask, None, vision_mask
                    )
                else:
                    encoded_features = self.model._encode_modalities(
                        text_tokens, None, None, vision_tensor,
                        text_mask, None, vision_mask
                    )
                
                # 融合特征
                fused_features = self.model.fusion_module(
                    text_features=encoded_features.get('text'),
                    audio_features=None,
                    vision_features=encoded_features.get('vision'),
                    text_mask=text_mask,
                    audio_mask=None,
                    vision_mask=vision_mask
                )
                
                generated_tokens = self.model._generate_text(
                    fused_features, max_length=max_length, temperature=temperature
                )
            
            # 解码响应
            response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # 更新对话历史
            input_desc = f"[{'视频' if is_video else '图像'}输入]" + (f" + {text_context}" if text_context else "")
            self._add_to_history(input_desc, response_text)
            
            return {
                'success': True,
                'response': response_text,
                'input_type': 'video' if is_video else 'image',
                'text_context': text_context,
                'metadata': {
                    'model_type': 'multimodal',
                    'vision_shape': list(image_or_video.shape),
                    'temperature': temperature
                }
            }
            
        except Exception as e:
            logger.error(f"视觉对话失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "抱歉，处理视觉输入时出现了错误。"
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
    
    def _add_to_history(self, user_input: str, ai_response: str):
        """添加到对话历史"""
        self.conversation_history.append({
            'user': user_input,
            'ai': ai_response,
            'timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else str(torch.randint(0, 1000000, (1,)).item())
        })
        
        # 限制历史长度
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MultiModalAI',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': len(self.tokenizer),
            'device': str(self.device),
            'config': self.config.to_dict()
        }
