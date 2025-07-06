"""
Flask服务器
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import numpy as np
import base64
import io
import logging
from typing import Dict, Any
import traceback

from .chat_api import ChatAPI

logger = logging.getLogger(__name__)


def create_app(model_path=None, config_path=None, device='auto'):
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 启用跨域支持
    
    # 初始化ChatAPI
    try:
        chat_api = ChatAPI(model_path=model_path, config_path=config_path, device=device)
        logger.info("ChatAPI初始化成功")
    except Exception as e:
        logger.error(f"ChatAPI初始化失败: {e}")
        chat_api = None
    
    @app.route('/')
    def index():
        """主页"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>多模态AI聊天系统</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .chat-box { border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
                .input-group { margin: 10px 0; }
                input, textarea, button { padding: 8px; margin: 5px; }
                button { background-color: #007bff; color: white; border: none; cursor: pointer; }
                button:hover { background-color: #0056b3; }
                .message { margin: 5px 0; padding: 5px; }
                .user-message { background-color: #e3f2fd; text-align: right; }
                .ai-message { background-color: #f1f8e9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 多模态AI聊天系统</h1>
                <div id="chat-box" class="chat-box"></div>
                
                <div class="input-group">
                    <textarea id="text-input" placeholder="输入您的消息..." style="width: 70%; height: 60px;"></textarea>
                    <button onclick="sendTextMessage()" style="height: 60px;">发送文本</button>
                </div>
                
                <div class="input-group">
                    <input type="file" id="audio-input" accept="audio/*">
                    <button onclick="sendAudioMessage()">发送音频</button>
                </div>
                
                <div class="input-group">
                    <input type="file" id="image-input" accept="image/*">
                    <button onclick="sendImageMessage()">发送图像</button>
                </div>
                
                <div class="input-group">
                    <button onclick="clearHistory()">清空历史</button>
                    <button onclick="getModelInfo()">模型信息</button>
                </div>
            </div>
            
            <script>
                function addMessage(sender, message) {
                    const chatBox = document.getElementById('chat-box');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'ai-message');
                    messageDiv.innerHTML = '<strong>' + sender + ':</strong> ' + message;
                    chatBox.appendChild(messageDiv);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
                
                function sendTextMessage() {
                    const textInput = document.getElementById('text-input');
                    const message = textInput.value.trim();
                    if (!message) return;
                    
                    addMessage('用户', message);
                    textInput.value = '';
                    
                    fetch('/chat/text', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: message})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            addMessage('AI', data.response);
                        } else {
                            addMessage('系统', '错误: ' + data.error);
                        }
                    })
                    .catch(error => {
                        addMessage('系统', '请求失败: ' + error);
                    });
                }
                
                function clearHistory() {
                    fetch('/chat/clear', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('chat-box').innerHTML = '';
                        addMessage('系统', '对话历史已清空');
                    });
                }
                
                function getModelInfo() {
                    fetch('/model/info')
                    .then(response => response.json())
                    .then(data => {
                        const info = JSON.stringify(data, null, 2);
                        addMessage('系统', '<pre>' + info + '</pre>');
                    });
                }
                
                // 回车发送消息
                document.getElementById('text-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendTextMessage();
                    }
                });
            </script>
        </body>
        </html>
        """
        return render_template_string(html_template)
    
    @app.route('/chat/text', methods=['POST'])
    def chat_text():
        """文本对话接口"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            data = request.get_json()
            text_input = data.get('text', '')
            
            if not text_input:
                return jsonify({'success': False, 'error': '文本输入不能为空'})
            
            # 获取可选参数
            max_length = data.get('max_length', 50)
            temperature = data.get('temperature', 1.0)
            top_k = data.get('top_k', 50)
            top_p = data.get('top_p', 0.9)
            
            result = chat_api.chat_text(
                text_input=text_input,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"文本对话接口错误: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    @app.route('/chat/audio', methods=['POST'])
    def chat_audio():
        """音频对话接口"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            data = request.get_json()
            
            # 解码音频数据
            audio_data_b64 = data.get('audio_data', '')
            if not audio_data_b64:
                return jsonify({'success': False, 'error': '音频数据不能为空'})
            
            audio_bytes = base64.b64decode(audio_data_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # 重塑为特征格式 [seq_len, feature_dim]
            feature_dim = data.get('feature_dim', 128)
            if len(audio_array) % feature_dim != 0:
                # 填充到合适的长度
                pad_length = feature_dim - (len(audio_array) % feature_dim)
                audio_array = np.pad(audio_array, (0, pad_length), 'constant')
            
            audio_features = audio_array.reshape(-1, feature_dim)
            
            text_context = data.get('text_context', None)
            max_length = data.get('max_length', 50)
            temperature = data.get('temperature', 1.0)
            
            result = chat_api.chat_audio(
                audio_features=audio_features,
                text_context=text_context,
                max_length=max_length,
                temperature=temperature
            )
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"音频对话接口错误: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    @app.route('/chat/vision', methods=['POST'])
    def chat_vision():
        """视觉对话接口"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            data = request.get_json()
            
            # 解码图像/视频数据
            vision_data_b64 = data.get('vision_data', '')
            if not vision_data_b64:
                return jsonify({'success': False, 'error': '视觉数据不能为空'})
            
            vision_bytes = base64.b64decode(vision_data_b64)
            vision_array = np.frombuffer(vision_bytes, dtype=np.float32)
            
            # 获取数据形状
            shape = data.get('shape', [3, 64, 64])  # 默认图像形状
            vision_tensor = torch.from_numpy(vision_array.reshape(shape))
            
            is_video = data.get('is_video', False)
            text_context = data.get('text_context', None)
            max_length = data.get('max_length', 50)
            temperature = data.get('temperature', 1.0)
            
            result = chat_api.chat_vision(
                image_or_video=vision_tensor,
                text_context=text_context,
                is_video=is_video,
                max_length=max_length,
                temperature=temperature
            )
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"视觉对话接口错误: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    @app.route('/chat/history', methods=['GET'])
    def get_history():
        """获取对话历史"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            history = chat_api.get_conversation_history()
            return jsonify({'success': True, 'history': history})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/chat/clear', methods=['POST'])
    def clear_history():
        """清空对话历史"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            chat_api.clear_conversation_history()
            return jsonify({'success': True, 'message': '对话历史已清空'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/model/info', methods=['GET'])
    def get_model_info():
        """获取模型信息"""
        if not chat_api:
            return jsonify({'success': False, 'error': 'ChatAPI未初始化'})
        
        try:
            info = chat_api.get_model_info()
            return jsonify(info)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查"""
        return jsonify({
            'status': 'healthy',
            'api_available': chat_api is not None
        })
    
    return app


def run_server(host='0.0.0.0', port=5000, debug=False, 
               model_path=None, config_path=None, device='auto'):
    """运行服务器"""
    app = create_app(model_path=model_path, config_path=config_path, device=device)
    
    logger.info(f"启动多模态AI聊天服务器")
    logger.info(f"地址: http://{host}:{port}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"配置路径: {config_path}")
    logger.info(f"设备: {device}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多模态AI聊天服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--model-path', help='模型文件路径')
    parser.add_argument('--config-path', help='配置文件路径')
    parser.add_argument('--device', default='auto', help='设备 (cpu/cuda/auto)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
