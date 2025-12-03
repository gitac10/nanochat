"""
AIME25评测任务：American Invitational Mathematics Examination 2025
这是一个数学竞赛数据集，包含需要数值答案的数学问题，实现了对aime24和25的评测
实现了本地评测和api在线评测两种方式
"""


import re
import json
from datasets import load_dataset
from tasks.common import Task
from openai import OpenAI

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

    def __init__(self, subset, split, data_path=None,dataset_type="aime25",use_llm = False, **kwargs):
        super().__init__(**kwargs)
        # AIME25只有主要的子集和标准分割
        assert subset in ["main", "train", "test"], "subset must be main|train|test"
        assert split in ["train", "dev", "test"], "split must be train|dev|test"
        self.dataset_type = dataset_type
        self.use_llm = use_llm
        ##默认加载方式
        self.ds = load_dataset("math-ai/aime25", split=split).shuffle(seed = 42)
        # ##本地加载
        # self.data_path = data_path or "D:/桌面/nanochat/AIME25/test.jsonl"  # 默认本地数据路径，支持 AIME25 目录
        # self.ds = load_dataset("json", data_files = self.data_path, split="train").shuffle(seed=42) 
        ##使用样例数据
        # self.ds = self._create_sample_data(split) 

        print(f"Loaded AIME25 {subset}/{split}: {len(self.ds)} examples")

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
                "problem": "what is the sum of 1 + 1",
                "answer": "2",
                "solution": "1+1=2"
            }
        ]
        return sample_data  

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
        # 方法1：尝试提取#### 后的数值
        marker_pattern = r"#### (\-?[0-9\.\,]+)"
        marker_match = re.search(marker_pattern, text)
        if marker_match:
            return marker_match.group(1)
        
        # 方法2：寻找答案标签格式 "答案是 X" 或 "answer is X"
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
        
        # 方法3：寻找最后一个出现的数字（适用于纯数字答案）
        numbers = re.findall(r'\d+', text)
        if numbers:
            return numbers[-1]  # 返回最后一个数字
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
        if self.dataset_type == 'aime25':
            reference_answer = str(row['answer'])
        else:
            solution_text = str(row['solution'])
            match = re.search(r'\\boxed\s*\{(\d+)\}', solution_text)
            
            # 2. 如果找到匹配项，使用括号内的数字；否则使用整个 solution 文本（作为兜底）
            reference_answer = match.group(1) if match else solution_text.strip()
        
        # 构建对话格式
        # 用户发送数学问题，助手给出参考答案和解答过程
        messages = [
            {
                "role": "user", 
                "content": (
                    f"Solve the following mathematics problem:\n\n"
                    f"{problem}\n\n"
                    f"Please provide your complete step-by-step reasoning first. "
                    f"Finally, state the numerical result immediately after the '####' marker."
                )
                # "content": f"Solve this mathematics problem:\n\n{problem}\n\n Remember to put your final answer after 'The answer is'."
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
        if self.use_llm:
            client = OpenAI(
                api_key="sk-gtaafqtnpyqqdwvjqvxmirkvmkudaaatkdqepyzarokoollm",
                base_url="https://api.siliconflow.cn/v1/"
            )

            # 提取真实答案和模型预测答案
            JUDGE_MODEL_NAME = "Qwen/Qwen3-32B" 
            
            # 提取真实答案和模型预测答案
            true_answer = conversation['reference_answer'].strip()
            predicted_answer = self.extract_answer(assistant_response).strip()
            
            # 提取原始问题内容 (从对话消息中获取)
            # 最后一个用户消息是实际问题
            problem = conversation['messages'][-1]['content'] 
            ##参考 opencompass 的判断模版，引入大模型进行判断
            
            GRADER_TEMPLATE = """
                Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
                Here are some evaluation criteria:
                1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
                2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
                3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
                4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
                5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
                6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
                A: CORRECT 
                B: INCORRECT
                C: INVALID
                Just return the letters "A", "B", or "C", with no text around it.
                Here is your task. Simply reply with either A, B, or C. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
                <Original Question Begin>:
                {question}
                <Original Question End>
                <Standard Answer Begin>:
                {answer}
                <Standard Answer End>
                <Candidate's Answer Begin>: 
                {prediction}
                <Candidate's Answer End>
                Judging the correctness of the candidate's answer:
                """
            
            # 3. 填充 Prompt
            judge_prompt = GRADER_TEMPLATE.format(
                question=problem,
                answer=true_answer,
                prediction=assistant_response
            )
            
            is_correct = 0 # 默认错误
            
            try:
                # 调用外部 API 服务进行裁判
                judge_response = client.chat.completions.create(
                    model=JUDGE_MODEL_NAME,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=20
                )
                
                judge_text = judge_response.choices[0].message.content.strip().upper()
                # 根据裁判模型的回复判断结果
                if "A" == judge_text:
                    is_correct = 1
                    
            except Exception as e:
                # API 调用失败，直接返回 0 (错误)，不进行任何回退。
                print(f"⚠️ 裁判 API 调用失败: {e}. 返回 0。")
                is_correct = 0
                
            print("大模型评判成功，判断结果为:", judge_text)
            return is_correct
        ##本地评判
        else:
        
            true_answer = conversation['reference_answer']
            predicted_answer = self.extract_answer(assistant_response)
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
        return float(self.evaluate(conversation, assistant_response))

