#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础文本对话示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.chat_api import ChatAPI


def main():
    """主函数"""
    print("🤖 多模态AI基础对话示例")
    print("=" * 50)
    
    # 模型路径（使用训练好的最佳模型）
    model_path = "checkpoints/best_model.pth"
    
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
        print(f"  词汇表大小: {model_info['vocab_size']}")
        print(f"  设备: {model_info['device']}")
        
        print("\n开始对话 (输入 'quit' 退出, 'clear' 清空历史):")
        print("-" * 50)
        
        while True:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            # 检查退出命令
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            # 检查清空历史命令
            if user_input.lower() == 'clear':
                chat_api.clear_conversation_history()
                print("对话历史已清空")
                continue
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 发送消息并获取回复
            try:
                result = chat_api.chat_text(
                    text_input=user_input,
                    max_length=50,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                
                if result['success']:
                    print(f"AI: {result['response']}")
                else:
                    print(f"错误: {result['error']}")
                    
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


if __name__ == '__main__':
    main()
