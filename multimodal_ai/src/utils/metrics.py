"""
评估指标工具
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import time


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                      ignore_index: int = -100) -> float:
    """计算准确率"""
    if ignore_index is not None:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
    
    if len(targets) == 0:
        return 0.0
    
    correct = (predictions == targets).sum().item()
    total = len(targets)
    return correct / total


def calculate_perplexity(loss: float) -> float:
    """计算困惑度"""
    return np.exp(loss)


def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """计算BLEU分数（简化版本）"""
    # 这里实现一个简化的BLEU分数计算
    # 实际应用中建议使用专业的BLEU计算库如sacrebleu
    
    if len(predictions) != len(references):
        raise ValueError("预测和参考序列数量不匹配")
    
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if len(pred_tokens) == 0:
            continue
        
        # 计算1-gram精确度
        pred_1gram = set(pred_tokens)
        ref_1gram = set(ref_tokens)
        
        if len(pred_1gram) == 0:
            precision = 0.0
        else:
            precision = len(pred_1gram & ref_1gram) / len(pred_1gram)
        
        # 简化的长度惩罚
        bp = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
        
        score = bp * precision
        total_score += score
    
    return total_score / len(predictions) if predictions else 0.0


def calculate_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算ROUGE分数（简化版本）"""
    if len(predictions) != len(references):
        raise ValueError("预测和参考序列数量不匹配")
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # ROUGE-1 (unigram)
        pred_1gram = set(pred_tokens)
        ref_1gram = set(ref_tokens)
        
        if len(ref_1gram) == 0:
            rouge_1 = 0.0
        else:
            rouge_1 = len(pred_1gram & ref_1gram) / len(ref_1gram)
        rouge_1_scores.append(rouge_1)
        
        # ROUGE-2 (bigram)
        pred_2gram = set(zip(pred_tokens[:-1], pred_tokens[1:]))
        ref_2gram = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        
        if len(ref_2gram) == 0:
            rouge_2 = 0.0
        else:
            rouge_2 = len(pred_2gram & ref_2gram) / len(ref_2gram)
        rouge_2_scores.append(rouge_2)
        
        # ROUGE-L (longest common subsequence)
        lcs_length = _lcs_length(pred_tokens, ref_tokens)
        if len(ref_tokens) == 0:
            rouge_l = 0.0
        else:
            rouge_l = lcs_length / len(ref_tokens)
        rouge_l_scores.append(rouge_l)
    
    return {
        'rouge_1': np.mean(rouge_1_scores),
        'rouge_2': np.mean(rouge_2_scores),
        'rouge_l': np.mean(rouge_l_scores)
    }


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """计算最长公共子序列长度"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def calculate_metrics(predictions: Union[torch.Tensor, List[str]], 
                     targets: Union[torch.Tensor, List[str]],
                     loss: Optional[float] = None,
                     metric_types: List[str] = None) -> Dict[str, float]:
    """计算多种评估指标"""
    
    if metric_types is None:
        metric_types = ['accuracy', 'perplexity']
    
    metrics = {}
    
    # 数值指标
    if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
        if 'accuracy' in metric_types:
            if predictions.dim() > 1:
                pred_labels = predictions.argmax(dim=-1)
            else:
                pred_labels = predictions
            metrics['accuracy'] = calculate_accuracy(pred_labels, targets)
        
        if 'perplexity' in metric_types and loss is not None:
            metrics['perplexity'] = calculate_perplexity(loss)
    
    # 文本指标
    if isinstance(predictions, list) and isinstance(targets, list):
        if 'bleu' in metric_types:
            metrics['bleu'] = calculate_bleu_score(predictions, targets)
        
        if any(metric.startswith('rouge') for metric in metric_types):
            rouge_scores = calculate_rouge_score(predictions, targets)
            for metric in metric_types:
                if metric in rouge_scores:
                    metrics[metric] = rouge_scores[metric]
    
    return metrics


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.step_count = 0
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """更新指标"""
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        for name, value in metrics.items():
            self.metrics[name].append((step, value, time.time()))
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """获取最新指标值"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1][1]
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> Optional[float]:
        """获取平均值"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = [item[1] for item in self.metrics[metric_name]]
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Optional[float]:
        """获取最佳值"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = [item[1] for item in self.metrics[metric_name]]
        if mode == 'max':
            return max(values)
        elif mode == 'min':
            return min(values)
        else:
            raise ValueError("mode必须是'max'或'min'")
    
    def get_history(self, metric_name: str) -> List[tuple]:
        """获取历史记录"""
        return self.metrics.get(metric_name, [])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标摘要"""
        summary = {}
        
        for metric_name in self.metrics:
            values = [item[1] for item in self.metrics[metric_name]]
            summary[metric_name] = {
                'latest': values[-1] if values else 0.0,
                'average': np.mean(values),
                'best_max': max(values),
                'best_min': min(values),
                'std': np.std(values),
                'count': len(values)
            }
        
        return summary
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.start_time = time.time()
        self.step_count = 0
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        import json
        
        data = {
            'metrics': dict(self.metrics),
            'start_time': self.start_time,
            'step_count': self.step_count
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """从文件加载"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metrics = defaultdict(list, data['metrics'])
        self.start_time = data['start_time']
        self.step_count = data['step_count']


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError("mode必须是'min'或'max'")
    
    def __call__(self, score: float, model: torch.nn.Module = None) -> bool:
        """检查是否应该早停"""
        
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best(self, model: torch.nn.Module):
        """恢复最佳权重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
