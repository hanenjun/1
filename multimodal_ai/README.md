# 🤖 MultiModal AI System

一个专业的多模态AI对话系统，支持文本对话、音频对话和视觉感知。

## ✨ 核心功能

- 📝 **文本对话**: 自然语言理解和生成
- 🎵 **音频对话**: 语音识别和语音合成
- 👁️ **视觉感知**: 图像理解和视频分析
- 🔄 **多模态融合**: 跨模态信息整合
- 🚀 **实时交互**: 低延迟响应系统

## 📁 项目结构

```
multimodal_ai/
├── src/                    # 源代码
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── text_model.py      # 文本处理模型
│   │   ├── audio_model.py     # 音频处理模型
│   │   ├── vision_model.py    # 视觉处理模型
│   │   ├── fusion_model.py    # 多模态融合模型
│   │   └── multimodal_ai.py   # 主模型
│   ├── data/              # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py         # 数据集类
│   │   ├── preprocessor.py    # 数据预处理
│   │   └── tokenizer.py       # 分词器
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py          # 日志工具
│   │   ├── metrics.py         # 评估指标
│   │   └── helpers.py         # 辅助函数
│   └── api/               # API接口
│       ├── __init__.py
│       ├── chat_api.py        # 对话API
│       └── server.py          # 服务器
├── config/                # 配置文件
│   ├── __init__.py
│   ├── base_config.py         # 基础配置
│   ├── model_config.py        # 模型配置
│   └── training_config.py     # 训练配置
├── tools/                 # 工具脚本
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── inference.py          # 推理脚本
│   └── export_model.py       # 模型导出
├── tests/                 # 测试用例
│   ├── __init__.py
│   ├── test_models.py        # 模型测试
│   ├── test_data.py          # 数据测试
│   └── test_api.py           # API测试
├── examples/              # 使用示例
│   ├── basic_chat.py         # 基础对话
│   ├── audio_chat.py         # 音频对话
│   └── vision_chat.py        # 视觉对话
├── docs/                  # 文档
│   ├── installation.md      # 安装指南
│   ├── usage.md             # 使用指南
│   └── api_reference.md     # API参考
├── checkpoints/           # 模型检查点
├── requirements.txt       # 依赖包
├── setup.py              # 安装脚本
└── README.md             # 项目说明
```

## 🚀 快速开始

### 安装
```bash
cd multimodal_ai
pip install -r requirements.txt
pip install -e .
```

### 基础使用
```python
from multimodal_ai import MultiModalAI

# 初始化AI系统
ai = MultiModalAI()

# 文本对话
response = ai.chat("你好，今天天气怎么样？")

# 音频对话
response = ai.chat_with_audio("audio_file.wav")

# 视觉感知
response = ai.chat_with_vision("image.jpg", "这张图片里有什么？")
```

### 训练模型
```bash
python tools/train.py --config config/training_config.py
```

### 启动API服务
```bash
python -m src.api.server --model-path checkpoints/model.pth --port 5000
```

### 命令行推理
```bash
# 交互式对话
python tools/inference.py --model-path checkpoints/model.pth --mode interactive

# 文本推理
python tools/inference.py --model-path checkpoints/model.pth --mode text --text "你好"

# 图像推理
python tools/inference.py --model-path checkpoints/model.pth --mode image --image-file image.jpg

# 音频推理
python tools/inference.py --model-path checkpoints/model.pth --mode audio --audio-file audio.wav
```

## 🌐 API接口

### 文本对话
```bash
curl -X POST http://localhost:5000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，请介绍一下自己"}'
```

### 音频对话
```bash
curl -X POST http://localhost:5000/chat/audio \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio", "text_context": "这是什么声音？"}'
```

### 视觉对话
```bash
curl -X POST http://localhost:5000/chat/vision \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64_encoded_image", "text_context": "描述这张图片"}'
```

### 模型信息
```bash
curl http://localhost:5000/model/info
```

## 📊 模型架构

### 核心组件
- **文本编码器**: 基于Transformer的文本编码，支持中文分词
- **音频编码器**: 卷积神经网络处理音频特征
- **视觉编码器**: 3D CNN处理视频，2D CNN处理图像
- **多模态融合**: 跨模态注意力机制整合多模态信息
- **文本解码器**: 自回归生成式解码器

### 模型参数
- **总参数量**: ~12.6M
- **文本编码器**: ~8M参数
- **音频编码器**: ~2M参数
- **视觉编码器**: ~2M参数
- **融合层**: ~0.6M参数

## 🔧 配置说明

主要配置文件位于 `config/model_config.py`：

```python
# 模型基础配置
vocab_size = 10000          # 词汇表大小
d_model = 512              # 模型维度
num_heads = 8              # 注意力头数
num_layers = 6             # 层数
max_seq_length = 128       # 最大序列长度

# 音频配置
audio_feature_dim = 128    # 音频特征维度
audio_seq_length = 100     # 音频序列长度

# 视觉配置
image_channels = 3         # 图像通道数
image_height = 64          # 图像高度
image_width = 64           # 图像宽度
video_frames = 8           # 视频帧数
```

## 🧪 测试

运行测试套件：
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_models.py
python -m pytest tests/test_api.py
```

## 🔄 模型导出

### 导出TorchScript
```bash
python tools/export_model.py \
    --model-path checkpoints/model.pth \
    --output-dir exports/ \
    --format torchscript
```

### 导出ONNX
```bash
python tools/export_model.py \
    --model-path checkpoints/model.pth \
    --output-dir exports/ \
    --format onnx
```

### 导出所有格式
```bash
python tools/export_model.py \
    --model-path checkpoints/model.pth \
    --output-dir exports/ \
    --format all
```

## 🛠️ 开发指南

详细的开发文档请参考 [docs/](docs/) 目录：
- [安装指南](docs/installation.md)
- [使用指南](docs/usage.md)
- [API参考](docs/api_reference.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- Transformer架构的原始论文作者
- 开源社区的贡献者们

---

🎯 **专业级多模态AI系统，支持文本、音频、视觉三大模态！**
