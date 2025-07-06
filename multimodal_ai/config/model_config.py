"""
模型配置类
"""

from .base_config import BaseConfig


class ModelConfig(BaseConfig):
    """模型配置类"""
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        # 模型基础配置
        self.model_name = "multimodal_ai"
        self.model_version = "1.0.0"
        
        # 文本模型配置
        self.vocab_size = 5000
        self.max_seq_length = 50
        self.text_embed_dim = 256
        
        # 音频模型配置
        self.audio_feature_dim = 128
        self.audio_seq_length = 100
        self.audio_embed_dim = 256
        
        # 视觉模型配置
        self.image_channels = 3
        self.image_height = 64
        self.image_width = 64
        self.video_frames = 8
        self.vision_embed_dim = 256
        
        # Transformer配置
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.dropout = 0.1
        self.activation = "relu"
        
        # 融合模型配置
        self.fusion_method = "attention"  # attention, concat, add
        self.fusion_hidden_dim = 512
        
        # 生成配置
        self.max_generate_length = 50
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.9
        self.repetition_penalty = 1.1
        
        # 特殊token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # 模型文件路径
        self.model_checkpoint = self.checkpoints_dir / "multimodal_ai_model.pth"
        self.tokenizer_path = self.checkpoints_dir / "tokenizer.json"
        
    def get_text_config(self):
        """获取文本模型配置"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.text_embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_encoder_layers,
            'max_seq_length': self.max_seq_length,
            'dropout': self.dropout
        }
    
    def get_audio_config(self):
        """获取音频模型配置"""
        return {
            'feature_dim': self.audio_feature_dim,
            'embed_dim': self.audio_embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_encoder_layers,
            'seq_length': self.audio_seq_length,
            'dropout': self.dropout
        }
    
    def get_vision_config(self):
        """获取视觉模型配置"""
        return {
            'channels': self.image_channels,
            'height': self.image_height,
            'width': self.image_width,
            'frames': self.video_frames,
            'embed_dim': self.vision_embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_encoder_layers,
            'dropout': self.dropout
        }
    
    def get_fusion_config(self):
        """获取融合模型配置"""
        return {
            'method': self.fusion_method,
            'input_dims': [self.text_embed_dim, self.audio_embed_dim, self.vision_embed_dim],
            'hidden_dim': self.fusion_hidden_dim,
            'output_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        }
    
    def get_decoder_config(self):
        """获取解码器配置"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_decoder_layers,
            'max_length': self.max_generate_length,
            'dropout': self.dropout
        }
