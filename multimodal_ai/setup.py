#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态AI对话系统安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# 版本信息
VERSION = "1.0.0"

setup(
    name="multimodal-ai",
    version=VERSION,
    author="AI Developer",
    author_email="developer@example.com",
    description="一个支持文本、音频、视觉感知的多模态人工智能对话系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/multimodal-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=21.7.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
        "export": [
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
        ],
        "full": [
            "black>=21.7.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-train=tools.train:main",
            "multimodal-evaluate=tools.evaluate:main",
            "multimodal-inference=tools.inference:main",
            "multimodal-export=tools.export_model:main",
            "multimodal-server=src.api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "multimodal",
        "deep learning",
        "natural language processing",
        "computer vision",
        "speech recognition",
        "transformer",
        "pytorch",
        "chinese",
        "conversation",
        "chatbot",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/multimodal-ai/issues",
        "Source": "https://github.com/your-username/multimodal-ai",
        "Documentation": "https://github.com/your-username/multimodal-ai/docs",
    },
)
