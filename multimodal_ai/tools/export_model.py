#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本
"""

import torch
import torch.onnx
import numpy as np
import logging
import sys
from pathlib import Path
import argparse
import json
import time
from typing import Dict, Any, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.multimodal_ai import MultiModalAI
from src.data.tokenizer import SimpleTokenizer
from config.model_config import ModelConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        if config_path:
            self.config = ModelConfig(config_path)
        else:
            self.config = ModelConfig()
        
        # 加载模型
        self.model = MultiModalAI(self.config)
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        
        self._load_checkpoint(model_path)
        
        logger.info(f"模型导出器初始化完成，设备: {self.device}")
    
    def _load_checkpoint(self, model_path: str):
        """加载检查点"""
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
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型已从 {model_path} 加载")
    
    def export_to_torchscript(self, output_path: str, trace_mode: bool = True):
        """导出为TorchScript格式"""
        logger.info(f"开始导出TorchScript模型到: {output_path}")
        
        # 创建示例输入
        batch_size = 1
        text_tokens = torch.randint(0, self.config.vocab_size, (batch_size, self.config.max_seq_length))
        audio_features = torch.randn(batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        video_frames = torch.randn(batch_size, self.config.image_channels, self.config.video_frames, 
                                  self.config.image_height, self.config.image_width)
        
        # 创建掩码
        text_mask = torch.ones(batch_size, self.config.max_seq_length)
        audio_mask = torch.ones(batch_size, self.config.audio_seq_length)
        vision_mask = torch.ones(batch_size, self.config.video_frames)
        
        # 移动到设备
        example_inputs = (
            text_tokens.to(self.device),
            audio_features.to(self.device),
            video_frames.to(self.device),
            None,  # image_frames
            text_mask.to(self.device),
            audio_mask.to(self.device),
            vision_mask.to(self.device),
            None   # target_mask
        )
        
        try:
            if trace_mode:
                # 使用trace模式
                traced_model = torch.jit.trace(self.model, example_inputs)
            else:
                # 使用script模式
                traced_model = torch.jit.script(self.model)
            
            # 保存模型
            traced_model.save(output_path)
            
            # 验证导出的模型
            loaded_model = torch.jit.load(output_path)
            with torch.no_grad():
                original_output = self.model(*example_inputs)
                traced_output = loaded_model(*example_inputs)
                
                # 检查输出是否一致
                if torch.allclose(original_output, traced_output, atol=1e-5):
                    logger.info("TorchScript模型验证成功")
                else:
                    logger.warning("TorchScript模型输出与原模型不一致")
            
            logger.info(f"TorchScript模型导出成功: {output_path}")
            
        except Exception as e:
            logger.error(f"TorchScript导出失败: {e}")
            raise
    
    def export_to_onnx(self, output_path: str, opset_version: int = 11):
        """导出为ONNX格式"""
        logger.info(f"开始导出ONNX模型到: {output_path}")
        
        # 创建示例输入
        batch_size = 1
        text_tokens = torch.randint(0, self.config.vocab_size, (batch_size, self.config.max_seq_length))
        audio_features = torch.randn(batch_size, self.config.audio_seq_length, self.config.audio_feature_dim)
        video_frames = torch.randn(batch_size, self.config.image_channels, self.config.video_frames,
                                  self.config.image_height, self.config.image_width)
        
        # 创建掩码
        text_mask = torch.ones(batch_size, self.config.max_seq_length)
        audio_mask = torch.ones(batch_size, self.config.audio_seq_length)
        vision_mask = torch.ones(batch_size, self.config.video_frames)
        
        # 移动到设备
        example_inputs = (
            text_tokens.to(self.device),
            audio_features.to(self.device),
            video_frames.to(self.device),
            None,  # image_frames
            text_mask.to(self.device),
            audio_mask.to(self.device),
            vision_mask.to(self.device),
            None   # target_mask
        )
        
        # 输入名称
        input_names = [
            'text_tokens',
            'audio_features', 
            'video_frames',
            'image_frames',
            'text_mask',
            'audio_mask',
            'vision_mask',
            'target_mask'
        ]
        
        # 输出名称
        output_names = ['logits']
        
        # 动态轴
        dynamic_axes = {
            'text_tokens': {0: 'batch_size', 1: 'seq_length'},
            'audio_features': {0: 'batch_size', 1: 'audio_length'},
            'video_frames': {0: 'batch_size', 2: 'video_length'},
            'text_mask': {0: 'batch_size', 1: 'seq_length'},
            'audio_mask': {0: 'batch_size', 1: 'audio_length'},
            'vision_mask': {0: 'batch_size', 1: 'video_length'},
            'logits': {0: 'batch_size', 1: 'output_length'}
        }
        
        try:
            torch.onnx.export(
                self.model,
                example_inputs,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            logger.info(f"ONNX模型导出成功: {output_path}")
            
            # 验证ONNX模型
            self._verify_onnx_model(output_path, example_inputs)
            
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise
    
    def _verify_onnx_model(self, onnx_path: str, example_inputs: Tuple):
        """验证ONNX模型"""
        try:
            import onnx
            import onnxruntime as ort
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话
            ort_session = ort.InferenceSession(onnx_path)
            
            # 准备输入数据
            ort_inputs = {}
            input_names = [input.name for input in ort_session.get_inputs()]
            
            for i, name in enumerate(input_names):
                if i < len(example_inputs) and example_inputs[i] is not None:
                    ort_inputs[name] = example_inputs[i].cpu().numpy()
            
            # 运行推理
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # 比较输出
            with torch.no_grad():
                torch_output = self.model(*example_inputs)
                torch_output_np = torch_output.cpu().numpy()
                
                if np.allclose(torch_output_np, ort_outputs[0], atol=1e-4):
                    logger.info("ONNX模型验证成功")
                else:
                    logger.warning("ONNX模型输出与原模型存在差异")
            
        except ImportError:
            logger.warning("未安装onnx或onnxruntime，跳过ONNX模型验证")
        except Exception as e:
            logger.warning(f"ONNX模型验证失败: {e}")
    
    def export_state_dict(self, output_path: str):
        """导出模型权重"""
        logger.info(f"开始导出模型权重到: {output_path}")
        
        export_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'tokenizer_vocab': {
                'word_to_id': self.tokenizer.word_to_id,
                'id_to_word': self.tokenizer.id_to_word
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'export_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        torch.save(export_data, output_path)
        logger.info(f"模型权重导出成功: {output_path}")
    
    def export_config(self, output_path: str):
        """导出配置文件"""
        logger.info(f"开始导出配置文件到: {output_path}")
        
        config_data = {
            'model_config': self.config.to_dict(),
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'vocab_size': len(self.tokenizer),
                'export_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'tokenizer_info': {
                'vocab_size': len(self.tokenizer),
                'special_tokens': {
                    'pad_token': self.config.pad_token,
                    'unk_token': self.config.unk_token,
                    'bos_token': self.config.bos_token,
                    'eos_token': self.config.eos_token
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置文件导出成功: {output_path}")
    
    def export_all(self, output_dir: str):
        """导出所有格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始导出所有格式到目录: {output_dir}")
        
        # 导出TorchScript
        try:
            torchscript_path = output_path / "model.pt"
            self.export_to_torchscript(str(torchscript_path))
        except Exception as e:
            logger.error(f"TorchScript导出失败: {e}")
        
        # 导出ONNX
        try:
            onnx_path = output_path / "model.onnx"
            self.export_to_onnx(str(onnx_path))
        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
        
        # 导出权重
        try:
            weights_path = output_path / "model_weights.pth"
            self.export_state_dict(str(weights_path))
        except Exception as e:
            logger.error(f"权重导出失败: {e}")
        
        # 导出配置
        try:
            config_path = output_path / "model_config.json"
            self.export_config(str(config_path))
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
        
        logger.info("所有格式导出完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='导出多模态AI模型')
    parser.add_argument('--model-path', required=True, help='模型文件路径')
    parser.add_argument('--config-path', help='配置文件路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    
    # 导出格式选择
    parser.add_argument('--format', choices=['torchscript', 'onnx', 'weights', 'config', 'all'],
                       default='all', help='导出格式')
    
    # TorchScript选项
    parser.add_argument('--trace-mode', action='store_true', help='使用trace模式导出TorchScript')
    
    # ONNX选项
    parser.add_argument('--opset-version', type=int, default=11, help='ONNX opset版本')
    
    args = parser.parse_args()
    
    # 创建导出器
    exporter = ModelExporter(args.model_path, args.config_path)
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 根据格式导出
    if args.format == 'torchscript':
        torchscript_path = output_path / "model.pt"
        exporter.export_to_torchscript(str(torchscript_path), args.trace_mode)
    
    elif args.format == 'onnx':
        onnx_path = output_path / "model.onnx"
        exporter.export_to_onnx(str(onnx_path), args.opset_version)
    
    elif args.format == 'weights':
        weights_path = output_path / "model_weights.pth"
        exporter.export_state_dict(str(weights_path))
    
    elif args.format == 'config':
        config_path = output_path / "model_config.json"
        exporter.export_config(str(config_path))
    
    elif args.format == 'all':
        exporter.export_all(args.output_dir)
    
    logger.info("模型导出完成！")


if __name__ == '__main__':
    main()
