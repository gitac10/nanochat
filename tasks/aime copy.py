"""
AIME25评测任务：American Invitational Mathematics Examination 2025
这是一个数学竞赛数据集，包含需要数值答案的数学问题
"""

import re
from datasets import load_dataset
from tasks.common import Task


class AIME(Task):
    """
    AIME25数学竞赛评测任务类
    
    AIME (American Invitational Mathematics Examination) 是一个高水平的数学竞赛，
    包含需要精确数值答案的代数、几何、数论、组合数学等问题。
    
    特点：
    - 答案通常是整数 (0-999 范围)
    - 需要精确的数学推理
    - 生成式评估类型
    - 支持多种子集和分割
    """

    def __init__(self, subset, split, data_path=None, **kwargs):
        super().__init__(**kwargs)
        # AIME25只有主要的子集和标准分割
        assert subset in ["main", "train", "test"], "subset must be main|train|test"
        assert split in ["train", "dev", "test"], "split must be train|dev|test"
        
        # 加载AIME25数据集
        self.data_path = data_path or "data/aime25"  # 默认本地数据路径
        self.ds = self._load_data(subset, split)
            
        print(f"Loaded AIME25 {subset}/{split}: {len(self.ds)} examples")
        
    def _load_data(self, subset, split):
        """
        智能数据加载：按优先级尝试不同数据源
        1. 本地数据集路径
        2. HuggingFace数据集
        3. 示例数据（兜底）
        """
        import os
        from datasets import load_from_disk
        
        # 方法1：尝试从本地加载
        local_data_path = self._get_local_data_path(subset, split)
        if local_data_path and os.path.exists(local_data_path):
            try:
                print(f"Loading local data from: {local_data_path}")
                ds = load_from_disk(local_data_path)
                return ds.shuffle(seed=42)
            except Exception as e:
                print(f"Failed to load local data: {e}")
        
        # 方法2：尝试从HuggingFace加载
        try:
            print("Attempting to load from HuggingFace...")
            ds = load_dataset("math-ai/aime25", subset, split=split)
            return ds.shuffle(seed=42)
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
        
        # 方法3：使用示例数据（兜底）
        print("Using sample data as fallback...")
        return self._create_sample_data(split)
    
    def _get_local_data_path(self, subset, split):
        """
        生成本地数据路径
        支持多种目录结构：
        1. data/aime25/{subset}_{split}/
        2. data/aime25/{split}/
        3. 自定义路径/data_path/aime25/{subset}_{split}/
        """
        if not self.data_path:
            return None
            
        import os
        
        # 尝试多种可能的路径结构
        possible_paths = [
            # 路径1: data/aime25/train/
            os.path.join(self.data_path, split),
            # 路径2: data/aime25/train.parquet
            os.path.join(self.data_path, f"{split}.parquet"),
            # 路径3: data/aime25/main_train/
            os.path.join(self.data_path, f"{subset}_{split}"),
            # 路径4: data/aime25/main_train.parquet
            os.path.join(self.data_path, f"{subset}_{split}.parquet"),
            # 路径5: data/aime25/train-00000-of-00001.parquet
            os.path.join(self.data_path, f"{split}-00000-of-00001.parquet"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found local data at: {path}")
                return path
        
        print(f"No local data found in: {self.data_path}")
        return None

    def _create_sample_data(self, split):
        """创建AIME25示例数据（实际使用时应从真实数据集加载）"""
        sample_data = [
            {
                "id": "AIME25_001",
                "problem": "What is the value of x in the equation 2x + 5 = 13?",
                "answer": "4",
                "solution": "Subtract 5 from both sides: 2x = 8, then divide by 2: x = 4"
            },
            {
                "id": "AIME25_002", 
                "problem": "If the area of a rectangle is 48 square units and its length is 12 units, what is its width?",
                "answer": "4",
                "solution": "Area = length × width, so 48 = 12 × width, thus width = 48 ÷ 12 = 4"
            },
            {
                "id": "AIME25_003",
                "problem": "What is the sum of the first 10 positive integers?",
                "answer": "55",
                "solution": "Sum = n(n+1)/2 = 10 × 11 / 2 = 110 / 2 = 55"
            },
            {
                "id": "AIME25_004",
                "problem": "If f(x) = 3x + 7, what is f(5)?",
                "answer": "22",
                "solution": "f(5) = 3 × 5 + 7 = 15 + 7 = 22"
            },
            {
                "id": "AIME25_005",
                "problem": "1 + 1 = ?",
                "answer": "2",
                "solution": "1+1=2"
            }
        ]
        
        # 根据split返回不同大小的子集
        if split == "train":
            return sample_data[:3]  # 前3个用于训练
        elif split == "dev":
            return sample_data[3:4]  # 第4个用于验证
        else:  # test
            return sample_data[0:]  # 最后一个用于测试

    @property
    def eval_type(self):
        """
        返回评测类型：AIME是生成式评测任务
        模型需要生成数学问题的数值答案
        """
        return 'generative'

    def num_examples(self):
        """返回数据集大小"""
        return len(self.ds)

    def extract_answer(self, text):
        """
        从模型输出中提取最终答案
        
        AIME的答案通常是整数，本函数会：
        1. 寻找数值答案
        2. 处理可能的推理过程
        3. 提取最终的数值结果
        
        Args:
            text (str): 模型生成的完整回答
            
        Returns:
            str: 提取的答案字符串
        """
        # 方法1：寻找答案标签格式 "答案是 X" 或 "answer is X"
        answer_patterns = [
            r'答案是\s*(\d+)',
            r'answer is\s*(\d+)',
            r'=\s*(\d+)',
            r'答案[:：]\s*(\d+)',
            r'Answer[:：]\s*(\d+)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # 方法2：寻找最后一个出现的数字（适用于纯数字答案）
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[-1]  # 返回最后一个数字
        
        # 方法3：尝试提取框起来的答案 [[answer]] 格式
        boxed_pattern = r'\[\[(\d+)\]\]'
        boxed_match = re.search(boxed_pattern, text)
        if boxed_match:
            return boxed_match.group(1)
            
        # 如果都无法提取，返回原文本
        return text.strip()

    def get_example(self, index):
        """
        获取单个AIME问题实例
        
        Args:
            index (int): 问题索引
            
        Returns:
            dict: 包含问题的对话格式数据
        """
        row = self.ds[index]
        problem = row['problem']
        reference_answer = str(row['answer'])
        
        # 构建对话格式
        # 用户发送数学问题，助手给出参考答案和解答过程
        messages = [
            {
                "role": "user", 
                "content": f"Solve this AIME mathematics problem:\n\n{problem}\n\nPlease provide your final numerical answer."
            },
            {
                "role": "assistant", 
                "content": f"Here is the solution:\n\n{row.get('solution', '')}\n\nThe answer is {reference_answer}."
            }
        ]
        
        conversation = {
            "messages": messages,
            "question_id": row.get('id', f"AIME_{index:03d}"),
            "reference_answer": reference_answer,  # 用于评估
            "problem_type": "AIME_mathematics"
        }
        
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估AIME问题答案的正确性
        
        Args:
            conversation (dict): 包含真实答案的完整对话
            assistant_response (str): 模型生成的答案
            
        Returns:
            int: 1表示正确，0表示错误
        """
        # 提取真实答案和模型预测答案
        true_answer = conversation['reference_answer']
        predicted_answer = self.extract_answer(assistant_response)
        print("*" * 30)
        print(f"True answer: {true_answer}")
        print(f"Predicted answer: {predicted_answer}")
        print("*" * 30)
        
        # AIME答案通常是整数，进行数值比较
        try:
            true_num = int(true_answer)
            pred_num = int(predicted_answer)
            is_correct = 1 if true_num == pred_num else 0
        except ValueError:
            # 如果无法转换为整数，使用字符串比较
            is_correct = 1 if predicted_answer.strip() == true_answer.strip() else 0
            
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        强化学习奖励函数
        
        对于AIME问题，提供更细粒度的奖励：
        - 完全正确：1.0
        - 部分正确（数值相近但不完全正确）：0.5
        - 错误：0.0
        
        Args:
            conversation (dict): 对话上下文
            assistant_response (str): 模型回答
            
        Returns:
            float: 奖励分数 (0.0-1.0)
        """
        true_answer = conversation['reference_answer']
        predicted_answer = self.extract_answer(assistant_response)
        
        try:
            true_num = int(true_answer)
            pred_num = int(predicted_answer)
            
            if true_num == pred_num:
                return 1.0
            elif abs(true_num - pred_num) <= 1:  # 允许±1的误差
                return 0.5
            else:
                return 0.0
        except ValueError:
            return 0.0


class AIME25WithSteps(AIME):
    """
    AIME25变体：需要展示解题步骤的版本
    
    这个版本评估模型不仅答案要正确，还需要展示合理的解题步骤
    """
    
    def evaluate(self, conversation, assistant_response):
        """
        增强评估：同时检查答案正确性和解题步骤的合理性
        
        Args:
            conversation (dict): 对话上下文
            assistant_response (str): 模型回答
            
        Returns:
            int: 1表示答案和步骤都正确，0表示其他情况
        """
        true_answer = conversation['reference_answer']
        predicted_answer = self.extract_answer(assistant_response)
        
        # 检查答案是否正确
        try:
            true_num = int(true_answer)
            pred_num = int(predicted_answer)
            answer_correct = (true_num == pred_num)
        except ValueError:
            answer_correct = (predicted_answer.strip() == true_answer.strip())
        
        # 检查解题步骤（简单检查：回答是否包含数学运算关键词）
        math_keywords = ['add', 'subtract', 'multiply', 'divide', '+', '-', '×', '÷', 
                        '加', '减', '乘', '除', '计算', '解', 'solve', 'calculate',
                        '因为', '所以', 'thus', 'therefore', 'hence']
        
        has_steps = any(keyword in assistant_response.lower() for keyword in math_keywords)
        
        # 完整评估：答案正确且有解题步骤
        if answer_correct and has_steps:
            return 1
        elif answer_correct:  # 答案正确但缺少步骤
            return 1  # AIME主要关注最终答案
        else:
            return 0