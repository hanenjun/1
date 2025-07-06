#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本
"""

import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_ai import MultiModalAI
from src.data.dataset import create_train_val_dataloaders
from src.api.chat_api import ChatAPI
from config.model_config import ModelConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        if config_path:
            self.config = ModelConfig(config_path)
        else:
            self.config = ModelConfig()
        
        # 初始化ChatAPI
        self.chat_api = ChatAPI(
            model_path=model_path,
            config_path=config_path,
            device=self.device
        )
        
        logger.info(f"评估器初始化完成，设备: {self.device}")
    
    def evaluate_text_generation(self, test_texts: List[str]) -> Dict[str, Any]:
        """评估文本生成能力"""
        logger.info("开始评估文本生成能力...")
        
        results = {
            'total_samples': len(test_texts),
            'successful_generations': 0,
            'failed_generations': 0,
            'average_response_time': 0,
            'responses': []
        }
        
        total_time = 0
        
        for i, text in enumerate(test_texts):
            logger.info(f"评估样本 {i+1}/{len(test_texts)}: {text[:50]}...")
            
            start_time = time.time()
            response = self.chat_api.chat_text(text)
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            if response['success']:
                results['successful_generations'] += 1
                results['responses'].append({
                    'input': text,
                    'output': response['response'],
                    'response_time': response_time,
                    'success': True
                })
            else:
                results['failed_generations'] += 1
                results['responses'].append({
                    'input': text,
                    'error': response.get('error', 'Unknown error'),
                    'response_time': response_time,
                    'success': False
                })
        
        results['average_response_time'] = total_time / len(test_texts)
        results['success_rate'] = results['successful_generations'] / results['total_samples']
        
        logger.info(f"文本生成评估完成:")
        logger.info(f"  成功率: {results['success_rate']:.2%}")
        logger.info(f"  平均响应时间: {results['average_response_time']:.3f}s")
        
        return results
    
    def evaluate_multimodal_capabilities(self) -> Dict[str, Any]:
        """评估多模态能力"""
        logger.info("开始评估多模态能力...")
        
        results = {
            'audio_processing': self._test_audio_processing(),
            'vision_processing': self._test_vision_processing(),
            'multimodal_fusion': self._test_multimodal_fusion()
        }
        
        return results
    
    def _test_audio_processing(self) -> Dict[str, Any]:
        """测试音频处理能力"""
        logger.info("测试音频处理...")
        
        # 生成测试音频特征
        test_audio = torch.randn(100, 128)  # [seq_len, feature_dim]
        
        try:
            start_time = time.time()
            response = self.chat_api.chat_audio(
                audio_features=test_audio,
                text_context="这是一段音频"
            )
            end_time = time.time()
            
            return {
                'success': response['success'],
                'response_time': end_time - start_time,
                'response': response.get('response', ''),
                'error': response.get('error', None)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0
            }
    
    def _test_vision_processing(self) -> Dict[str, Any]:
        """测试视觉处理能力"""
        logger.info("测试视觉处理...")
        
        # 生成测试图像
        test_image = torch.randn(3, 64, 64)  # [channels, height, width]
        
        try:
            start_time = time.time()
            response = self.chat_api.chat_vision(
                image_or_video=test_image,
                text_context="这是一张图片",
                is_video=False
            )
            end_time = time.time()
            
            return {
                'success': response['success'],
                'response_time': end_time - start_time,
                'response': response.get('response', ''),
                'error': response.get('error', None)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0
            }
    
    def _test_multimodal_fusion(self) -> Dict[str, Any]:
        """测试多模态融合能力"""
        logger.info("测试多模态融合...")
        
        # 生成测试视频
        test_video = torch.randn(3, 8, 64, 64)  # [channels, frames, height, width]
        
        try:
            start_time = time.time()
            response = self.chat_api.chat_vision(
                image_or_video=test_video,
                text_context="描述这个视频内容",
                is_video=True
            )
            end_time = time.time()
            
            return {
                'success': response['success'],
                'response_time': end_time - start_time,
                'response': response.get('response', ''),
                'error': response.get('error', None)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': 0
            }
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """评估模型性能"""
        logger.info("开始评估模型性能...")
        
        model_info = self.chat_api.get_model_info()
        
        # 测试推理速度
        test_text = "你好，请介绍一下自己。"
        inference_times = []
        
        for i in range(10):
            start_time = time.time()
            self.chat_api.chat_text(test_text)
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        performance_metrics = {
            'model_info': model_info,
            'inference_speed': {
                'average_time': sum(inference_times) / len(inference_times),
                'min_time': min(inference_times),
                'max_time': max(inference_times),
                'std_time': torch.tensor(inference_times).std().item()
            },
            'memory_usage': self._get_memory_usage()
        }
        
        return performance_metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'gpu_max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        else:
            return {'gpu_memory': 'N/A (CPU only)'}
    
    def run_comprehensive_evaluation(self, output_file: str = None) -> Dict[str, Any]:
        """运行综合评估"""
        logger.info("开始综合评估...")
        
        # 测试文本
        test_texts = [
            "你好，请介绍一下自己。",
            "今天天气怎么样？",
            "请解释一下人工智能的概念。",
            "你能帮我写一首诗吗？",
            "什么是深度学习？"
        ]
        
        evaluation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'text_generation': self.evaluate_text_generation(test_texts),
            'multimodal_capabilities': self.evaluate_multimodal_capabilities(),
            'model_performance': self.evaluate_model_performance()
        }
        
        # 计算总体评分
        text_score = evaluation_results['text_generation']['success_rate']
        audio_score = 1.0 if evaluation_results['multimodal_capabilities']['audio_processing']['success'] else 0.0
        vision_score = 1.0 if evaluation_results['multimodal_capabilities']['vision_processing']['success'] else 0.0
        fusion_score = 1.0 if evaluation_results['multimodal_capabilities']['multimodal_fusion']['success'] else 0.0
        
        overall_score = (text_score + audio_score + vision_score + fusion_score) / 4
        evaluation_results['overall_score'] = overall_score
        
        logger.info(f"综合评估完成，总体评分: {overall_score:.2%}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果保存到: {output_file}")
        
        return evaluation_results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("模型评估摘要")
        print("="*60)
        
        print(f"评估时间: {results['timestamp']}")
        print(f"总体评分: {results['overall_score']:.2%}")
        
        print("\n文本生成能力:")
        text_results = results['text_generation']
        print(f"  成功率: {text_results['success_rate']:.2%}")
        print(f"  平均响应时间: {text_results['average_response_time']:.3f}s")
        
        print("\n多模态能力:")
        multimodal = results['multimodal_capabilities']
        print(f"  音频处理: {'✓' if multimodal['audio_processing']['success'] else '✗'}")
        print(f"  视觉处理: {'✓' if multimodal['vision_processing']['success'] else '✗'}")
        print(f"  多模态融合: {'✓' if multimodal['multimodal_fusion']['success'] else '✗'}")
        
        print("\n模型性能:")
        performance = results['model_performance']
        model_info = performance['model_info']
        print(f"  参数量: {model_info['total_parameters']:,}")
        print(f"  平均推理时间: {performance['inference_speed']['average_time']:.3f}s")
        
        if 'gpu_memory_allocated' in performance['memory_usage']:
            print(f"  GPU内存使用: {performance['memory_usage']['gpu_memory_allocated']:.2f}GB")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估多模态AI模型')
    parser.add_argument('--model-path', required=True, help='模型文件路径')
    parser.add_argument('--config-path', help='配置文件路径')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    # 运行评估
    results = evaluator.run_comprehensive_evaluation(args.output)
    
    # 打印摘要
    evaluator.print_evaluation_summary(results)


if __name__ == '__main__':
    main()
