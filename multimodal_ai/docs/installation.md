# 安装指南

本文档将指导您完成多模态AI对话系统的安装和配置。

## 系统要求

### 硬件要求
- **CPU**: 现代多核处理器 (推荐 Intel i5 或 AMD Ryzen 5 以上)
- **内存**: 最少 8GB RAM (推荐 16GB 或更多)
- **存储**: 至少 5GB 可用空间
- **GPU**: 可选，支持 CUDA 的 NVIDIA GPU (推荐用于训练)

### 软件要求
- **操作系统**: 
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS 10.14+
  - Windows 10+
- **Python**: 3.8 - 3.11
- **CUDA**: 11.0+ (如果使用GPU)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-username/multimodal-ai.git
cd multimodal-ai
```

### 2. 创建虚拟环境

#### 使用 conda (推荐)
```bash
conda create -n multimodal-ai python=3.9
conda activate multimodal-ai
```

#### 使用 venv
```bash
python -m venv multimodal-ai-env
source multimodal-ai-env/bin/activate  # Linux/macOS
# 或
multimodal-ai-env\Scripts\activate     # Windows
```

### 3. 安装依赖

#### 基础安装
```bash
pip install -r requirements.txt
```

#### 开发环境安装
```bash
pip install -e ".[dev]"
```

#### 完整安装 (包含所有可选依赖)
```bash
pip install -e ".[full]"
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## GPU 支持配置

### NVIDIA GPU (CUDA)

1. **安装 CUDA Toolkit**
   - 访问 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)
   - 下载并安装适合您系统的 CUDA 版本

2. **验证 CUDA 安装**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **安装 PyTorch GPU 版本**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### AMD GPU (ROCm) - Linux 专用

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## 配置文件设置

### 1. 创建配置目录
```bash
mkdir -p config
mkdir -p checkpoints
mkdir -p data
mkdir -p logs
```

### 2. 复制默认配置
```bash
cp config/model_config.py.example config/model_config.py
cp config/training_config.py.example config/training_config.py
```

### 3. 编辑配置文件
根据您的硬件配置调整参数：

```python
# config/model_config.py
class ModelConfig:
    # 根据GPU内存调整批次大小
    batch_size = 16 if torch.cuda.is_available() else 4
    
    # 根据需要调整模型大小
    d_model = 512  # 可选: 256, 512, 768, 1024
    num_layers = 6  # 可选: 4, 6, 8, 12
```

## 数据准备

### 1. 创建数据目录结构
```bash
mkdir -p data/{train,val,test}
mkdir -p data/{audio,images,videos}
```

### 2. 下载预训练模型 (可选)
```bash
# 下载预训练的词向量
wget https://example.com/pretrained_embeddings.bin -O data/embeddings.bin

# 下载预训练模型检查点
wget https://example.com/pretrained_model.pth -O checkpoints/pretrained_model.pth
```

## 快速测试

### 1. 运行单元测试
```bash
python -m pytest tests/ -v
```

### 2. 测试模型加载
```bash
python -c "
from src.models.multimodal_ai import MultiModalAI
from config.model_config import ModelConfig
config = ModelConfig()
model = MultiModalAI(config)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
"
```

### 3. 测试数据生成
```bash
python tools/generate_sample_data.py --num-samples 100 --output-dir data/samples
```

## 常见问题解决

### 1. CUDA 内存不足
```python
# 在配置文件中减少批次大小
batch_size = 8  # 或更小

# 启用梯度检查点
gradient_checkpointing = True
```

### 2. 依赖冲突
```bash
# 清理环境重新安装
pip uninstall torch torchvision torchaudio
pip cache purge
pip install -r requirements.txt
```

### 3. 权限问题 (Linux/macOS)
```bash
# 确保有写入权限
chmod -R 755 multimodal-ai/
chown -R $USER:$USER multimodal-ai/
```

### 4. Windows 路径问题
```python
# 在 Windows 上使用原始字符串
model_path = r"C:\path\to\model.pth"
# 或使用正斜杠
model_path = "C:/path/to/model.pth"
```

## 性能优化

### 1. 编译优化
```bash
# 启用 PyTorch 编译优化
export TORCH_COMPILE=1
```

### 2. 内存优化
```python
# 在训练配置中启用
mixed_precision = True
gradient_accumulation_steps = 4
```

### 3. 多GPU 训练
```bash
# 使用 DataParallel
python tools/train.py --multi-gpu

# 使用 DistributedDataParallel
torchrun --nproc_per_node=2 tools/train.py --distributed
```

## 下一步

安装完成后，您可以：

1. 查看 [使用指南](usage.md) 了解如何使用系统
2. 查看 [API 参考](api_reference.md) 了解详细的API文档
3. 运行示例程序开始体验多模态对话功能

## 获取帮助

如果遇到问题，请：

1. 查看 [FAQ](faq.md)
2. 搜索 [GitHub Issues](https://github.com/your-username/multimodal-ai/issues)
3. 提交新的 Issue 描述您的问题

---

🎉 **恭喜！您已成功安装多模态AI对话系统！**
