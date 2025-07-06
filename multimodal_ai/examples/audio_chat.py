#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频对话示例
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.chat_api import ChatAPI


def generate_mock_audio_features(duration_seconds=2.0, sample_rate=16000, feature_dim=128):
    """
    生成模拟音频特征
    
    Args:
        duration_seconds: 音频时长（秒）
        sample_rate: 采样率
        feature_dim: 特征维度
    
    Returns:
        torch.Tensor: 音频特征张量
    """
    # 计算特征序列长度（假设每10ms一个特征）
    seq_length = int(duration_seconds * 100)  # 100 features per second
    
    # 生成模拟的音频特征
    # 这里使用随机数据，实际应用中应该是从音频文件提取的特征
    audio_features = torch.randn(seq_length, feature_dim)
    
    # 添加一些模式使其更像真实音频特征
    # 添加低频成分
    for i in range(0, seq_length, 10):
        audio_features[i:i+5] += torch.sin(torch.linspace(0, 2*np.pi, 5)).unsqueeze(1) * 0.5
    
    return audio_features


def simulate_different_audio_types():
    """模拟不同类型的音频"""
    audio_types = {
        "音乐": {
            "features": generate_mock_audio_features(3.0, feature_dim=128),
            "description": "这是一段音乐"
        },
        "语音": {
            "features": generate_mock_audio_features(2.0, feature_dim=128),
            "description": "这是一段语音"
        },
        "环境音": {
            "features": generate_mock_audio_features(4.0, feature_dim=128),
            "description": "这是环境声音"
        },
        "动物叫声": {
            "features": generate_mock_audio_features(1.5, feature_dim=128),
            "description": "这是动物的叫声"
        }
    }
    return audio_types


def main():
    """主函数"""
    print("🎵 多模态AI音频对话示例")
    print("=" * 50)
    
    # 模型路径
    model_path = "checkpoints/multimodal_chat_model.pth"
    
    try:
        # 初始化聊天API
        print("正在加载模型...")
        chat_api = ChatAPI(model_path)
        print("模型加载完成！")
        
        # 显示模型信息
        model_info = chat_api.get_model_info()
        print(f"\n模型信息:")
        print(f"  模型名称: {model_info['model_name']}")
        print(f"  参数量: {model_info['total_parameters']:,}")
        print(f"  设备: {model_info['device']}")
        
        # 生成不同类型的音频样本
        audio_samples = simulate_different_audio_types()
        
        print("\n可用的音频样本:")
        for i, (audio_type, _) in enumerate(audio_samples.items(), 1):
            print(f"  {i}. {audio_type}")
        
        print("\n开始音频对话 (输入数字选择音频类型, 'quit' 退出):")
        print("-" * 50)
        
        while True:
            # 获取用户选择
            user_input = input("\n请选择音频类型 (1-4) 或输入问题: ").strip()
            
            # 检查退出命令
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            try:
                # 检查是否是数字选择
                if user_input.isdigit():
                    choice = int(user_input)
                    if 1 <= choice <= len(audio_samples):
                        audio_type = list(audio_samples.keys())[choice - 1]
                        audio_data = audio_samples[audio_type]
                        
                        print(f"\n正在分析 {audio_type} 音频...")
                        
                        # 发送音频数据进行分析
                        result = chat_api.chat_audio(
                            audio_features=audio_data["features"],
                            text_context=f"请分析这段音频，这是{audio_data['description']}",
                            max_length=100,
                            temperature=0.7
                        )
                        
                        if result['success']:
                            print(f"AI分析: {result['response']}")
                        else:
                            print(f"错误: {result['error']}")
                    else:
                        print("无效选择，请输入1-4之间的数字")
                
                else:
                    # 用户输入的是问题，随机选择一个音频样本
                    audio_type = list(audio_samples.keys())[0]  # 默认使用第一个
                    audio_data = audio_samples[audio_type]
                    
                    print(f"\n正在结合音频回答问题...")
                    
                    result = chat_api.chat_audio(
                        audio_features=audio_data["features"],
                        text_context=user_input,
                        max_length=100,
                        temperature=0.8
                    )
                    
                    if result['success']:
                        print(f"AI回复: {result['response']}")
                    else:
                        print(f"错误: {result['error']}")
                        
            except ValueError:
                print("请输入有效的数字或问题")
            except Exception as e:
                print(f"发生错误: {e}")
        
        # 显示对话历史
        history = chat_api.get_conversation_history()
        if history:
            print(f"\n本次对话共 {len(history)} 轮:")
            for i, item in enumerate(history, 1):
                print(f"{i}. 用户: {item['user']}")
                print(f"   AI: {item['ai']}")
    
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行训练脚本生成模型文件")
    except Exception as e:
        print(f"初始化失败: {e}")


def demo_audio_processing():
    """演示音频处理功能"""
    print("\n🔊 音频处理演示")
    print("-" * 30)
    
    # 生成不同长度的音频特征
    short_audio = generate_mock_audio_features(1.0)
    medium_audio = generate_mock_audio_features(3.0)
    long_audio = generate_mock_audio_features(5.0)
    
    print(f"短音频特征形状: {short_audio.shape}")
    print(f"中等音频特征形状: {medium_audio.shape}")
    print(f"长音频特征形状: {long_audio.shape}")
    
    # 显示特征统计信息
    print(f"\n音频特征统计:")
    print(f"  均值: {short_audio.mean():.4f}")
    print(f"  标准差: {short_audio.std():.4f}")
    print(f"  最小值: {short_audio.min():.4f}")
    print(f"  最大值: {short_audio.max():.4f}")


if __name__ == '__main__':
    # 运行主程序
    main()
    
    # 运行音频处理演示
    demo_audio_processing()
