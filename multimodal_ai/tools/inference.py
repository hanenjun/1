#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理脚本
"""

import torch
import numpy as np
import logging
import sys
from pathlib import Path
import argparse
import json
import time
from typing import Optional, Union

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.chat_api import ChatAPI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'auto'):
        self.chat_api = ChatAPI(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        logger.info("推理引擎初始化完成")
    
    def text_inference(self, 
                      text: str,
                      max_length: int = 50,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.9) -> dict:
        """文本推理"""
        logger.info(f"文本推理: {text[:50]}...")
        
        start_time = time.time()
        result = self.chat_api.chat_text(
            text_input=text,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        end_time = time.time()
        
        result['inference_time'] = end_time - start_time
        return result
    
    def audio_inference(self,
                       audio_file: str,
                       text_context: Optional[str] = None,
                       max_length: int = 50,
                       temperature: float = 1.0) -> dict:
        """音频推理"""
        logger.info(f"音频推理: {audio_file}")
        
        # 这里应该加载真实的音频文件并提取特征
        # 目前使用随机特征作为示例
        audio_features = torch.randn(100, 128)  # [seq_len, feature_dim]
        
        start_time = time.time()
        result = self.chat_api.chat_audio(
            audio_features=audio_features,
            text_context=text_context,
            max_length=max_length,
            temperature=temperature
        )
        end_time = time.time()
        
        result['inference_time'] = end_time - start_time
        return result
    
    def vision_inference(self,
                        image_file: str,
                        text_context: Optional[str] = None,
                        is_video: bool = False,
                        max_length: int = 50,
                        temperature: float = 1.0) -> dict:
        """视觉推理"""
        logger.info(f"视觉推理: {image_file}")
        
        # 这里应该加载真实的图像/视频文件
        # 目前使用随机数据作为示例
        if is_video:
            vision_data = torch.randn(3, 8, 64, 64)  # [channels, frames, height, width]
        else:
            vision_data = torch.randn(3, 64, 64)  # [channels, height, width]
        
        start_time = time.time()
        result = self.chat_api.chat_vision(
            image_or_video=vision_data,
            text_context=text_context,
            is_video=is_video,
            max_length=max_length,
            temperature=temperature
        )
        end_time = time.time()
        
        result['inference_time'] = end_time - start_time
        return result
    
    def interactive_chat(self):
        """交互式聊天"""
        print("🤖 多模态AI聊天系统")
        print("输入 'quit' 退出，'clear' 清空历史，'help' 查看帮助")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    self.chat_api.clear_conversation_history()
                    print("对话历史已清空")
                    continue
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif not user_input:
                    continue
                
                # 检查是否是特殊命令
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # 普通文本对话
                result = self.text_inference(user_input)
                
                if result['success']:
                    print(f"AI: {result['response']}")
                    print(f"(推理时间: {result['inference_time']:.3f}s)")
                else:
                    print(f"错误: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    def _handle_command(self, command: str):
        """处理特殊命令"""
        parts = command[1:].split()
        cmd = parts[0].lower()
        
        if cmd == 'audio':
            if len(parts) < 2:
                print("用法: /audio <音频文件路径> [文本上下文]")
                return
            
            audio_file = parts[1]
            text_context = ' '.join(parts[2:]) if len(parts) > 2 else None
            
            result = self.audio_inference(audio_file, text_context)
            if result['success']:
                print(f"AI: {result['response']}")
                print(f"(推理时间: {result['inference_time']:.3f}s)")
            else:
                print(f"错误: {result['error']}")
        
        elif cmd == 'image':
            if len(parts) < 2:
                print("用法: /image <图像文件路径> [文本上下文]")
                return
            
            image_file = parts[1]
            text_context = ' '.join(parts[2:]) if len(parts) > 2 else None
            
            result = self.vision_inference(image_file, text_context, is_video=False)
            if result['success']:
                print(f"AI: {result['response']}")
                print(f"(推理时间: {result['inference_time']:.3f}s)")
            else:
                print(f"错误: {result['error']}")
        
        elif cmd == 'video':
            if len(parts) < 2:
                print("用法: /video <视频文件路径> [文本上下文]")
                return
            
            video_file = parts[1]
            text_context = ' '.join(parts[2:]) if len(parts) > 2 else None
            
            result = self.vision_inference(video_file, text_context, is_video=True)
            if result['success']:
                print(f"AI: {result['response']}")
                print(f"(推理时间: {result['inference_time']:.3f}s)")
            else:
                print(f"错误: {result['error']}")
        
        elif cmd == 'history':
            history = self.chat_api.get_conversation_history()
            if history:
                print("\n对话历史:")
                for i, item in enumerate(history, 1):
                    print(f"{i}. 用户: {item['user']}")
                    print(f"   AI: {item['ai']}")
            else:
                print("暂无对话历史")
        
        elif cmd == 'info':
            info = self.chat_api.get_model_info()
            print(f"\n模型信息:")
            print(f"  模型名称: {info['model_name']}")
            print(f"  参数量: {info['total_parameters']:,}")
            print(f"  词汇表大小: {info['vocab_size']}")
            print(f"  设备: {info['device']}")
        
        else:
            print(f"未知命令: {cmd}")
            self._print_help()
    
    def _print_help(self):
        """打印帮助信息"""
        help_text = """
可用命令:
  quit          - 退出程序
  clear         - 清空对话历史
  help          - 显示此帮助信息
  /audio <文件> [上下文] - 音频推理
  /image <文件> [上下文] - 图像推理
  /video <文件> [上下文] - 视频推理
  /history      - 查看对话历史
  /info         - 查看模型信息

示例:
  你好，请介绍一下自己
  /audio audio.wav 这是什么声音？
  /image photo.jpg 描述这张图片
  /video video.mp4 这个视频在做什么？
        """
        print(help_text)
    
    def batch_inference(self, input_file: str, output_file: str):
        """批量推理"""
        logger.info(f"开始批量推理: {input_file} -> {output_file}")
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            inputs = json.load(f)
        
        results = []
        
        for i, item in enumerate(inputs):
            logger.info(f"处理第 {i+1}/{len(inputs)} 个样本")
            
            input_type = item.get('type', 'text')
            
            if input_type == 'text':
                result = self.text_inference(
                    text=item['text'],
                    max_length=item.get('max_length', 50),
                    temperature=item.get('temperature', 1.0)
                )
            elif input_type == 'audio':
                result = self.audio_inference(
                    audio_file=item['audio_file'],
                    text_context=item.get('text_context'),
                    max_length=item.get('max_length', 50),
                    temperature=item.get('temperature', 1.0)
                )
            elif input_type == 'image':
                result = self.vision_inference(
                    image_file=item['image_file'],
                    text_context=item.get('text_context'),
                    is_video=False,
                    max_length=item.get('max_length', 50),
                    temperature=item.get('temperature', 1.0)
                )
            elif input_type == 'video':
                result = self.vision_inference(
                    image_file=item['video_file'],
                    text_context=item.get('text_context'),
                    is_video=True,
                    max_length=item.get('max_length', 50),
                    temperature=item.get('temperature', 1.0)
                )
            else:
                result = {'success': False, 'error': f'未知输入类型: {input_type}'}
            
            results.append({
                'input': item,
                'output': result
            })
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量推理完成，结果保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态AI推理')
    parser.add_argument('--model-path', required=True, help='模型文件路径')
    parser.add_argument('--config-path', help='配置文件路径')
    parser.add_argument('--device', default='auto', help='设备 (cpu/cuda/auto)')
    
    # 推理模式
    parser.add_argument('--mode', choices=['interactive', 'text', 'audio', 'image', 'video', 'batch'],
                       default='interactive', help='推理模式')
    
    # 文本推理参数
    parser.add_argument('--text', help='输入文本')
    parser.add_argument('--max-length', type=int, default=50, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--top-k', type=int, default=50, help='Top-K采样')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-P采样')
    
    # 多模态推理参数
    parser.add_argument('--audio-file', help='音频文件路径')
    parser.add_argument('--image-file', help='图像文件路径')
    parser.add_argument('--video-file', help='视频文件路径')
    parser.add_argument('--text-context', help='文本上下文')
    
    # 批量推理参数
    parser.add_argument('--input-file', help='批量推理输入文件')
    parser.add_argument('--output-file', help='批量推理输出文件')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    engine = InferenceEngine(args.model_path, args.config_path, args.device)
    
    # 根据模式执行推理
    if args.mode == 'interactive':
        engine.interactive_chat()
    
    elif args.mode == 'text':
        if not args.text:
            print("错误: 文本模式需要 --text 参数")
            return
        
        result = engine.text_inference(
            text=args.text,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        if result['success']:
            print(f"输入: {args.text}")
            print(f"输出: {result['response']}")
            print(f"推理时间: {result['inference_time']:.3f}s")
        else:
            print(f"错误: {result['error']}")
    
    elif args.mode == 'audio':
        if not args.audio_file:
            print("错误: 音频模式需要 --audio-file 参数")
            return
        
        result = engine.audio_inference(
            audio_file=args.audio_file,
            text_context=args.text_context,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        if result['success']:
            print(f"音频文件: {args.audio_file}")
            print(f"文本上下文: {args.text_context or '无'}")
            print(f"输出: {result['response']}")
            print(f"推理时间: {result['inference_time']:.3f}s")
        else:
            print(f"错误: {result['error']}")
    
    elif args.mode == 'image':
        if not args.image_file:
            print("错误: 图像模式需要 --image-file 参数")
            return
        
        result = engine.vision_inference(
            image_file=args.image_file,
            text_context=args.text_context,
            is_video=False,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        if result['success']:
            print(f"图像文件: {args.image_file}")
            print(f"文本上下文: {args.text_context or '无'}")
            print(f"输出: {result['response']}")
            print(f"推理时间: {result['inference_time']:.3f}s")
        else:
            print(f"错误: {result['error']}")
    
    elif args.mode == 'video':
        if not args.video_file:
            print("错误: 视频模式需要 --video-file 参数")
            return
        
        result = engine.vision_inference(
            image_file=args.video_file,
            text_context=args.text_context,
            is_video=True,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        if result['success']:
            print(f"视频文件: {args.video_file}")
            print(f"文本上下文: {args.text_context or '无'}")
            print(f"输出: {result['response']}")
            print(f"推理时间: {result['inference_time']:.3f}s")
        else:
            print(f"错误: {result['error']}")
    
    elif args.mode == 'batch':
        if not args.input_file or not args.output_file:
            print("错误: 批量模式需要 --input-file 和 --output-file 参数")
            return
        
        engine.batch_inference(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
