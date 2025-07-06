#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API测试
"""

import pytest
import torch
import json
import base64
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.chat_api import ChatAPI
from config.model_config import ModelConfig


class TestChatAPI:
    """测试聊天API"""
    
    def setup_method(self):
        """设置测试"""
        self.config = ModelConfig()
        
        # 创建模拟的模型路径
        self.mock_model_path = "mock_model.pth"
        
        # 模拟模型和分词器
        with patch('src.api.chat_api.MultiModalAI') as mock_model, \
             patch('src.api.chat_api.SimpleTokenizer') as mock_tokenizer, \
             patch('torch.load') as mock_load:
            
            # 设置模拟返回值
            mock_load.return_value = {
                'model_state_dict': {},
                'tokenizer_vocab': {
                    'word_to_id': {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3},
                    'id_to_word': {'0': '<pad>', '1': '<unk>', '2': '<bos>', '3': '<eos>'}
                }
            }
            
            # 创建API实例
            self.api = ChatAPI(self.mock_model_path)
    
    def test_api_initialization(self):
        """测试API初始化"""
        assert self.api is not None
        assert hasattr(self.api, 'model')
        assert hasattr(self.api, 'tokenizer')
        assert hasattr(self.api, 'config')
        assert hasattr(self.api, 'device')
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.api.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'vocab_size' in info
        assert 'device' in info
        
        # 检查数据类型
        assert isinstance(info['model_name'], str)
        assert isinstance(info['total_parameters'], int)
        assert isinstance(info['vocab_size'], int)
        assert isinstance(info['device'], str)
    
    def test_conversation_history(self):
        """测试对话历史管理"""
        # 初始状态应该为空
        history = self.api.get_conversation_history()
        assert history == []
        
        # 添加对话
        self.api._add_to_history("用户输入", "AI回复")
        history = self.api.get_conversation_history()
        assert len(history) == 1
        assert history[0]['user'] == "用户输入"
        assert history[0]['ai'] == "AI回复"
        
        # 清空历史
        self.api.clear_conversation_history()
        history = self.api.get_conversation_history()
        assert history == []
    
    @patch('src.api.chat_api.ChatAPI._generate_text')
    def test_chat_text(self, mock_generate):
        """测试文本聊天"""
        # 设置模拟返回值
        mock_generate.return_value = "这是AI的回复"
        
        # 测试文本聊天
        result = self.api.chat_text("你好")
        
        assert result['success'] is True
        assert 'response' in result
        assert result['response'] == "这是AI的回复"
        
        # 检查历史记录
        history = self.api.get_conversation_history()
        assert len(history) == 1
    
    @patch('src.api.chat_api.ChatAPI._generate_text')
    def test_chat_audio(self, mock_generate):
        """测试音频聊天"""
        # 设置模拟返回值
        mock_generate.return_value = "这是对音频的回复"
        
        # 创建模拟音频特征
        audio_features = torch.randn(100, 128)
        
        # 测试音频聊天
        result = self.api.chat_audio(audio_features, "这是什么声音？")
        
        assert result['success'] is True
        assert 'response' in result
        assert result['response'] == "这是对音频的回复"
    
    @patch('src.api.chat_api.ChatAPI._generate_text')
    def test_chat_vision(self, mock_generate):
        """测试视觉聊天"""
        # 设置模拟返回值
        mock_generate.return_value = "这是对图像的描述"
        
        # 创建模拟图像数据
        image_data = torch.randn(3, 64, 64)
        
        # 测试视觉聊天
        result = self.api.chat_vision(image_data, "描述这张图片", is_video=False)
        
        assert result['success'] is True
        assert 'response' in result
        assert result['response'] == "这是对图像的描述"
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空输入
        result = self.api.chat_text("")
        assert result['success'] is False
        assert 'error' in result
        
        # 测试None输入
        result = self.api.chat_text(None)
        assert result['success'] is False
        assert 'error' in result


class TestAPIServer:
    """测试API服务器"""
    
    def setup_method(self):
        """设置测试"""
        # 这里我们需要模拟Flask应用
        with patch('src.api.server.ChatAPI') as mock_api:
            from src.api.server import app
            self.app = app
            self.client = app.test_client()
            
            # 设置测试模式
            app.config['TESTING'] = True
    
    def test_health_check(self):
        """测试健康检查"""
        response = self.client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'
    
    @patch('src.api.server.chat_api')
    def test_text_chat_endpoint(self, mock_api):
        """测试文本聊天端点"""
        # 设置模拟返回值
        mock_api.chat_text.return_value = {
            'success': True,
            'response': '你好！我是AI助手。'
        }
        
        # 发送POST请求
        response = self.client.post(
            '/chat/text',
            data=json.dumps({'text': '你好'}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'response' in data
    
    @patch('src.api.server.chat_api')
    def test_audio_chat_endpoint(self, mock_api):
        """测试音频聊天端点"""
        # 设置模拟返回值
        mock_api.chat_audio.return_value = {
            'success': True,
            'response': '我听到了音频内容。'
        }
        
        # 创建模拟音频数据
        audio_data = np.random.randn(1000).astype(np.float32)
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        
        # 发送POST请求
        response = self.client.post(
            '/chat/audio',
            data=json.dumps({
                'audio_data': audio_base64,
                'text_context': '这是什么声音？'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'response' in data
    
    @patch('src.api.server.chat_api')
    def test_vision_chat_endpoint(self, mock_api):
        """测试视觉聊天端点"""
        # 设置模拟返回值
        mock_api.chat_vision.return_value = {
            'success': True,
            'response': '我看到了图像内容。'
        }
        
        # 创建模拟图像数据
        image_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image_base64 = base64.b64encode(image_data.tobytes()).decode('utf-8')
        
        # 发送POST请求
        response = self.client.post(
            '/chat/vision',
            data=json.dumps({
                'image_data': image_base64,
                'text_context': '描述这张图片',
                'is_video': False
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'response' in data
    
    @patch('src.api.server.chat_api')
    def test_model_info_endpoint(self, mock_api):
        """测试模型信息端点"""
        # 设置模拟返回值
        mock_api.get_model_info.return_value = {
            'model_name': 'MultiModalAI',
            'total_parameters': 12600000,
            'vocab_size': 10000,
            'device': 'cpu'
        }
        
        # 发送GET请求
        response = self.client.get('/model/info')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_name' in data
        assert 'total_parameters' in data
        assert 'vocab_size' in data
        assert 'device' in data
    
    def test_invalid_request(self):
        """测试无效请求"""
        # 测试缺少必需字段的请求
        response = self.client.post(
            '/chat/text',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        
        # 测试无效的JSON
        response = self.client.post(
            '/chat/text',
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_cors_headers(self):
        """测试CORS头"""
        response = self.client.get('/')
        
        # 检查CORS头是否存在
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers


class TestAPIIntegration:
    """测试API集成"""
    
    @pytest.mark.integration
    def test_full_conversation_flow(self):
        """测试完整对话流程"""
        # 这个测试需要真实的模型，通常在集成测试中运行
        pass
    
    @pytest.mark.integration
    def test_multimodal_interaction(self):
        """测试多模态交互"""
        # 这个测试需要真实的模型和数据，通常在集成测试中运行
        pass


if __name__ == '__main__':
    pytest.main([__file__])
