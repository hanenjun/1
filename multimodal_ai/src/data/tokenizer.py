"""
分词器模块
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class SimpleTokenizer:
    """简单的分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.id_to_word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.next_id = 4
        
        # 特殊token
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        word_freq = {}
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序，取前vocab_size-4个词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:self.vocab_size-4]:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单的空格分词，可以扩展为更复杂的分词逻辑
        return text.lower().strip().split()
    
    def encode(self, text: str, max_length: int = 50, add_special_tokens: bool = True) -> torch.Tensor:
        """编码文本"""
        words = self._tokenize(text)
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_token_id)  # BOS token
            max_content_length = max_length - 2  # 为BOS和EOS预留空间
        else:
            max_content_length = max_length
        
        # 编码词汇
        for word in words[:max_content_length]:
            tokens.append(self.word_to_id.get(word, self.unk_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)  # EOS token
        
        # 填充到固定长度
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)  # PAD token
            
        return torch.tensor(tokens[:max_length], dtype=torch.long)
    
    def decode(self, tokens: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """解码tokens"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        words = []
        for token in tokens:
            if skip_special_tokens and token in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                if token == self.eos_token_id:
                    break  # 遇到EOS停止解码
                continue
            
            word = self.id_to_word.get(token, self.unk_token)
            if word != self.unk_token or not skip_special_tokens:
                words.append(word)
        
        return ' '.join(words)
    
    def batch_encode(self, texts: List[str], max_length: int = 50, 
                    add_special_tokens: bool = True) -> torch.Tensor:
        """批量编码"""
        encoded_texts = []
        for text in texts:
            encoded_texts.append(self.encode(text, max_length, add_special_tokens))
        return torch.stack(encoded_texts)
    
    def batch_decode(self, batch_tokens: torch.Tensor, 
                    skip_special_tokens: bool = True) -> List[str]:
        """批量解码"""
        decoded_texts = []
        for tokens in batch_tokens:
            decoded_texts.append(self.decode(tokens, skip_special_tokens))
        return decoded_texts
    
    def save_vocab(self, save_path: Union[str, Path]):
        """保存词汇表"""
        save_path = Path(save_path)
        vocab_data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'next_id': self.next_id,
            'special_tokens': {
                'pad_token': self.pad_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
                'unk_token': self.unk_token
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, load_path: Union[str, Path]):
        """加载词汇表"""
        load_path = Path(load_path)
        
        with open(load_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.word_to_id = vocab_data['word_to_id']
        # 确保id_to_word的键是整数
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        self.next_id = vocab_data['next_id']
        
        # 加载特殊token（如果存在）
        if 'special_tokens' in vocab_data:
            special_tokens = vocab_data['special_tokens']
            self.pad_token = special_tokens.get('pad_token', '<PAD>')
            self.bos_token = special_tokens.get('bos_token', '<BOS>')
            self.eos_token = special_tokens.get('eos_token', '<EOS>')
            self.unk_token = special_tokens.get('unk_token', '<UNK>')
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.word_to_id)
    
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.word_to_id)
    
    def __repr__(self) -> str:
        return f"SimpleTokenizer(vocab_size={len(self.word_to_id)})"


class ChineseTokenizer(SimpleTokenizer):
    """中文分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        super().__init__(vocab_size)
        
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 简单的字符级分词，可以集成jieba等分词工具
        text = text.strip()
        tokens = []
        
        i = 0
        while i < len(text):
            char = text[i]
            if char.isspace():
                i += 1
                continue
            elif char.isascii() and char.isalpha():
                # 英文单词
                word = ''
                while i < len(text) and text[i].isascii() and text[i].isalpha():
                    word += text[i].lower()
                    i += 1
                if word:
                    tokens.append(word)
            else:
                # 中文字符或其他字符
                tokens.append(char)
                i += 1
        
        return tokens
