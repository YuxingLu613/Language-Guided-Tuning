import random
import re
import nltk
from nltk.corpus import wordnet
import torch

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextAugmenter:
    """文本数据增强类，提供多种文本增强方法"""
    
    def __init__(self, random_state=None):
        """
        初始化文本增强器
        
        Args:
            random_state: 随机种子，用于复现结果
        """
        if random_state is not None:
            random.seed(random_state)
            torch.manual_seed(random_state)
        
    def synonym_replacement(self, text, n=1):
        """
        随机替换n个非停用词为其同义词
        
        Args:
            text: 输入文本
            n: 要替换的词数量
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.3)))  # 最多替换30%的词
        
        # 随机选择n个词进行替换
        random_word_indices = random.sample(range(len(words)), n)
        
        for i in random_word_indices:
            word = words[i]
            synonyms = self._get_synonyms(word)
            if synonyms:
                words[i] = random.choice(synonyms)
                
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        """
        以概率p随机删除词语
        
        Args:
            text: 输入文本
            p: 每个词被删除的概率
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        # 保留的词
        kept_words = []
        for word in words:
            if random.random() > p:
                kept_words.append(word)
                
        # 确保至少保留一个词
        if len(kept_words) == 0:
            return random.choice(words)
            
        return ' '.join(kept_words)
    
    def random_swap(self, text, n=1):
        """
        随机交换文本中的n对词语
        
        Args:
            text: 输入文本
            n: 要交换的词对数量
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.3)))  # 最多交换30%的词对
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def random_insertion(self, text, n=1):
        """
        随机在文本中插入n个同义词
        
        Args:
            text: 输入文本
            n: 要插入的词数量
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        n = min(n, max(1, int(len(words) * 0.3)))  # 最多插入30%的词
        
        for _ in range(n):
            # 随机选择一个词
            random_idx = random.randint(0, len(words) - 1)
            random_word = words[random_idx]
            
            # 获取同义词
            synonyms = self._get_synonyms(random_word)
            if not synonyms:
                continue
                
            # 随机选择一个同义词并插入
            synonym = random.choice(synonyms)
            insert_idx = random.randint(0, len(words))
            words.insert(insert_idx, synonym)
            
        return ' '.join(words)
    
    def back_translation(self, text):
        """
        模拟回译效果（简化版）
        实际应用中可以使用翻译API或预训练模型
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        # 简化版回译，随机替换一些常见词语模拟翻译不准确的效果
        common_replacements = {
            'good': ['nice', 'great', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'slight'],
            'happy': ['glad', 'pleased', 'delighted'],
            'sad': ['unhappy', 'sorrowful', 'depressed'],
            'beautiful': ['pretty', 'lovely', 'attractive'],
            'ugly': ['unattractive', 'hideous', 'unsightly'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['sluggish', 'unhurried', 'leisurely'],
        }
        
        words = text.split()
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in common_replacements:
                words[i] = random.choice(common_replacements[lower_word])
                
        return ' '.join(words)
    
    def _get_synonyms(self, word):
        """获取词语的同义词"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms
    
    def apply_augmentation(self, text, aug_type, **kwargs):
        """
        应用指定的增强方法
        
        Args:
            text: 输入文本
            aug_type: 增强方法类型
            **kwargs: 增强方法的参数
            
        Returns:
            增强后的文本
        """
        if aug_type == "synonym_replacement":
            n = kwargs.get('n', 1)
            return self.synonym_replacement(text, n)
        elif aug_type == "random_deletion":
            p = kwargs.get('p', 0.1)
            return self.random_deletion(text, p)
        elif aug_type == "random_swap":
            n = kwargs.get('n', 1)
            return self.random_swap(text, n)
        elif aug_type == "random_insertion":
            n = kwargs.get('n', 1)
            return self.random_insertion(text, n)
        elif aug_type == "back_translation":
            return self.back_translation(text)
        elif aug_type == "none" or aug_type is None:
            return text
        else:
            raise ValueError(f"不支持的增强方法: {aug_type}")
