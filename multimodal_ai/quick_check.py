#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查脚本 - 验证项目结构完整性
"""

import sys
from pathlib import Path

def check_file_structure():
    """检查文件结构"""
    print("📁 检查项目文件结构...")
    
    required_files = [
        # 核心模型文件
        "src/models/__init__.py",
        "src/models/multimodal_ai.py",
        "src/models/text_model.py",
        "src/models/audio_model.py",
        "src/models/vision_model.py",
        "src/models/fusion_model.py",
        
        # 数据处理文件
        "src/data/__init__.py",
        "src/data/tokenizer.py",
        "src/data/dataset.py",
        "src/data/preprocessor.py",
        
        # API文件
        "src/api/__init__.py",
        "src/api/chat_api.py",
        "src/api/server.py",
        
        # 工具文件
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "src/utils/metrics.py",
        "src/utils/helpers.py",
        
        # 配置文件
        "config/__init__.py",
        "config/base_config.py",
        "config/model_config.py",
        "config/training_config.py",
        
        # 工具脚本
        "tools/__init__.py",
        "tools/train.py",
        "tools/evaluate.py",
        "tools/inference.py",
        "tools/export_model.py",
        
        # 测试文件
        "tests/__init__.py",
        "tests/test_models.py",
        "tests/test_data.py",
        "tests/test_api.py",
        
        # 示例文件
        "examples/__init__.py",
        "examples/basic_chat.py",
        "examples/audio_chat.py",
        "examples/vision_chat.py",
        
        # 项目文件
        "requirements.txt",
        "setup.py",
        "README.md",
        "verify_installation.py",
        
        # 文档文件
        "docs/installation.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n📊 文件统计:")
    print(f"  存在文件: {len(existing_files)}")
    print(f"  缺失文件: {len(missing_files)}")
    print(f"  完整度: {len(existing_files)}/{len(required_files)} ({len(existing_files)/len(required_files)*100:.1f}%)")
    
    if missing_files:
        print(f"\n❌ 缺失的文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ 所有必需文件都存在")
    return True


def check_directory_structure():
    """检查目录结构"""
    print("\n📂 检查目录结构...")
    
    required_dirs = [
        "src",
        "src/models",
        "src/data", 
        "src/api",
        "src/utils",
        "config",
        "tools",
        "tests",
        "examples",
        "docs",
        "checkpoints",  # 可能不存在，但应该可以创建
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            if dir_path == "checkpoints":
                print(f"ℹ️ {dir_path}/ (将在训练时创建)")
            else:
                print(f"❌ {dir_path}/")
                return False
    
    return True


def check_python_syntax():
    """检查Python语法"""
    print("\n🐍 检查Python语法...")
    
    python_files = [
        "src/models/multimodal_ai.py",
        "src/models/text_model.py", 
        "src/models/audio_model.py",
        "src/models/vision_model.py",
        "src/models/fusion_model.py",
        "src/data/tokenizer.py",
        "src/data/dataset.py",
        "src/data/preprocessor.py",
        "src/api/chat_api.py",
        "src/api/server.py",
        "config/model_config.py",
        "tools/train.py",
        "examples/basic_chat.py"
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, file_path, 'exec')
                print(f"✅ {file_path}")
            except SyntaxError as e:
                print(f"❌ {file_path}: 语法错误 - {e}")
                syntax_errors.append((file_path, str(e)))
            except Exception as e:
                print(f"⚠️ {file_path}: 其他错误 - {e}")
        else:
            print(f"⚠️ {file_path}: 文件不存在")
    
    if syntax_errors:
        print(f"\n❌ 发现 {len(syntax_errors)} 个语法错误:")
        for file_path, error in syntax_errors:
            print(f"  {file_path}: {error}")
        return False
    
    return True


def check_imports():
    """检查基本导入（不需要安装依赖）"""
    print("\n📦 检查基本导入结构...")
    
    # 只检查标准库导入
    try:
        import sys
        import os
        import json
        import logging
        import random
        from pathlib import Path
        from typing import Dict, List, Optional, Tuple, Any, Union
        print("✅ 标准库导入正常")
        
        # 检查项目内部导入结构（不实际导入）
        config_files = [
            "config/base_config.py",
            "config/model_config.py", 
            "config/training_config.py"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"✅ {config_file} 存在")
            else:
                print(f"❌ {config_file} 不存在")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 标准库导入失败: {e}")
        return False


def check_requirements():
    """检查requirements.txt"""
    print("\n📋 检查依赖配置...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt 不存在")
        return False
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read().strip().split('\n')
        
        required_packages = [
            'torch', 'torchvision', 'torchaudio',
            'numpy', 'pillow', 'opencv-python',
            'librosa', 'flask', 'flask-cors'
        ]
        
        found_packages = []
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                package_name = req.split('>=')[0].split('==')[0].split('[')[0].strip()
                found_packages.append(package_name.lower())
        
        missing_packages = []
        for package in required_packages:
            if package.lower() not in found_packages:
                missing_packages.append(package)
        
        print(f"✅ requirements.txt 包含 {len(found_packages)} 个依赖")
        
        if missing_packages:
            print(f"⚠️ 可能缺少的关键依赖: {missing_packages}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取requirements.txt失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 多模态AI项目快速检查")
    print("=" * 50)
    
    checks = [
        ("目录结构", check_directory_structure),
        ("文件结构", check_file_structure),
        ("Python语法", check_python_syntax),
        ("基本导入", check_imports),
        ("依赖配置", check_requirements),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            print(f"\n{'='*20} {check_name} {'='*20}")
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} 检查时发生错误: {e}")
            results.append((check_name, False))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📋 检查结果总结:")
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 项目结构检查完全通过！")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行完整验证: python verify_installation.py")
        print("3. 开始训练模型: python tools/train.py")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 项检查未通过")
        print("请根据上述信息修复问题后重新检查")
        return 1


if __name__ == '__main__':
    sys.exit(main())
