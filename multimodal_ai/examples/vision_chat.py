#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉对话示例
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.chat_api import ChatAPI


def generate_mock_image(width=64, height=64, channels=3, image_type="random"):
    """
    生成模拟图像数据
    
    Args:
        width: 图像宽度
        height: 图像高度
        channels: 通道数
        image_type: 图像类型
    
    Returns:
        torch.Tensor: 图像张量
    """
    if image_type == "gradient":
        # 生成渐变图像
        image = torch.zeros(channels, height, width)
        for i in range(height):
            for j in range(width):
                image[:, i, j] = torch.tensor([i/height, j/width, (i+j)/(height+width)])
    
    elif image_type == "checkerboard":
        # 生成棋盘图像
        image = torch.zeros(channels, height, width)
        for i in range(height):
            for j in range(width):
                if (i // 8 + j // 8) % 2 == 0:
                    image[:, i, j] = 1.0
    
    elif image_type == "circles":
        # 生成圆形图案
        image = torch.zeros(channels, height, width)
        center_x, center_y = width // 2, height // 2
        for i in range(height):
            for j in range(width):
                distance = ((i - center_y) ** 2 + (j - center_x) ** 2) ** 0.5
                if distance < min(width, height) // 4:
                    image[:, i, j] = 1.0
    
    else:  # random
        # 生成随机图像
        image = torch.randn(channels, height, width)
        image = torch.clamp(image, 0, 1)
    
    return image


def generate_mock_video(frames=8, width=64, height=64, channels=3, video_type="moving_circle"):
    """
    生成模拟视频数据
    
    Args:
        frames: 帧数
        width: 视频宽度
        height: 视频高度
        channels: 通道数
        video_type: 视频类型
    
    Returns:
        torch.Tensor: 视频张量
    """
    video = torch.zeros(channels, frames, height, width)
    
    if video_type == "moving_circle":
        # 生成移动圆形视频
        for frame in range(frames):
            # 圆心位置随时间变化
            center_x = int(width * (0.2 + 0.6 * frame / frames))
            center_y = height // 2
            
            for i in range(height):
                for j in range(width):
                    distance = ((i - center_y) ** 2 + (j - center_x) ** 2) ** 0.5
                    if distance < min(width, height) // 6:
                        video[:, frame, i, j] = 1.0
    
    elif video_type == "rotating_pattern":
        # 生成旋转图案视频
        center_x, center_y = width // 2, height // 2
        for frame in range(frames):
            angle = 2 * np.pi * frame / frames
            for i in range(height):
                for j in range(width):
                    # 旋转坐标
                    x = j - center_x
                    y = i - center_y
                    rotated_x = x * np.cos(angle) - y * np.sin(angle)
                    rotated_y = x * np.sin(angle) + y * np.cos(angle)
                    
                    if abs(rotated_x) < 5 or abs(rotated_y) < 5:
                        video[:, frame, i, j] = 1.0
    
    else:  # random
        # 生成随机视频
        video = torch.randn(channels, frames, height, width)
        video = torch.clamp(video, 0, 1)
    
    return video


def simulate_different_visual_content():
    """模拟不同类型的视觉内容"""
    visual_content = {
        "风景图片": {
            "data": generate_mock_image(image_type="gradient"),
            "type": "image",
            "description": "这是一张风景图片"
        },
        "几何图案": {
            "data": generate_mock_image(image_type="checkerboard"),
            "type": "image",
            "description": "这是几何图案"
        },
        "圆形图像": {
            "data": generate_mock_image(image_type="circles"),
            "type": "image",
            "description": "这是包含圆形的图像"
        },
        "移动物体视频": {
            "data": generate_mock_video(video_type="moving_circle"),
            "type": "video",
            "description": "这是一个移动物体的视频"
        },
        "旋转图案视频": {
            "data": generate_mock_video(video_type="rotating_pattern"),
            "type": "video",
            "description": "这是一个旋转图案的视频"
        }
    }
    return visual_content


def main():
    """主函数"""
    print("👁️ 多模态AI视觉对话示例")
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
        
        # 生成不同类型的视觉内容
        visual_samples = simulate_different_visual_content()
        
        print("\n可用的视觉内容:")
        for i, (content_type, info) in enumerate(visual_samples.items(), 1):
            print(f"  {i}. {content_type} ({info['type']})")
        
        print("\n开始视觉对话 (输入数字选择内容, 'quit' 退出):")
        print("-" * 50)
        
        while True:
            # 获取用户选择
            user_input = input("\n请选择视觉内容 (1-5) 或输入问题: ").strip()
            
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
                    if 1 <= choice <= len(visual_samples):
                        content_name = list(visual_samples.keys())[choice - 1]
                        content_data = visual_samples[content_name]
                        
                        print(f"\n正在分析 {content_name}...")
                        
                        # 发送视觉数据进行分析
                        result = chat_api.chat_vision(
                            visual_data=content_data["data"],
                            text_context=f"请分析这个{content_data['type']}，{content_data['description']}",
                            is_video=(content_data["type"] == "video"),
                            max_length=100,
                            temperature=0.7
                        )
                        
                        if result['success']:
                            print(f"AI分析: {result['response']}")
                        else:
                            print(f"错误: {result['error']}")
                    else:
                        print("无效选择，请输入1-5之间的数字")
                
                else:
                    # 用户输入的是问题，随机选择一个视觉内容
                    content_name = list(visual_samples.keys())[0]  # 默认使用第一个
                    content_data = visual_samples[content_name]
                    
                    print(f"\n正在结合视觉内容回答问题...")
                    
                    result = chat_api.chat_vision(
                        visual_data=content_data["data"],
                        text_context=user_input,
                        is_video=(content_data["type"] == "video"),
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


def demo_visual_processing():
    """演示视觉处理功能"""
    print("\n📸 视觉处理演示")
    print("-" * 30)
    
    # 生成不同类型的视觉内容
    image = generate_mock_image(64, 64, 3, "gradient")
    video = generate_mock_video(8, 64, 64, 3, "moving_circle")
    
    print(f"图像形状: {image.shape}")
    print(f"视频形状: {video.shape}")
    
    # 显示数据统计信息
    print(f"\n图像统计:")
    print(f"  均值: {image.mean():.4f}")
    print(f"  标准差: {image.std():.4f}")
    print(f"  最小值: {image.min():.4f}")
    print(f"  最大值: {image.max():.4f}")
    
    print(f"\n视频统计:")
    print(f"  均值: {video.mean():.4f}")
    print(f"  标准差: {video.std():.4f}")
    print(f"  最小值: {video.min():.4f}")
    print(f"  最大值: {video.max():.4f}")
    
    # 分析视频中的运动
    frame_diff = torch.abs(video[:, 1:] - video[:, :-1]).mean()
    print(f"  帧间差异: {frame_diff:.4f}")


def demo_multimodal_interaction():
    """演示多模态交互"""
    print("\n🔄 多模态交互演示")
    print("-" * 30)
    
    # 创建包含文本、图像的多模态输入
    text_context = "请描述这张图片中的内容"
    image_data = generate_mock_image(image_type="circles")
    
    print(f"文本输入: {text_context}")
    print(f"图像数据形状: {image_data.shape}")
    
    # 这里可以展示如何将多种模态结合使用
    print("多模态输入已准备就绪，可以发送给AI模型进行处理")


if __name__ == '__main__':
    # 运行主程序
    main()
    
    # 运行视觉处理演示
    demo_visual_processing()
    
    # 运行多模态交互演示
    demo_multimodal_interaction()
