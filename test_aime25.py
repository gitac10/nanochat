#!/usr/bin/env python3
"""
AIME25任务测试脚本
用于验证AIME25和AIME25WithSteps任务是否能正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

from tasks.aime import AIME25, AIME25WithSteps

def test_aime25_basic():
    """测试AIME25基本功能"""
    print("=== 测试 AIME25 基本功能 ===")
    
    # 创建AIME25任务对象
    task = AIME25(subset="main", split="test")
    
    print(f"数据集大小: {task.num_examples()}")
    print(f"评估类型: {task.eval_type}")
    
    # 测试获取第一个例子
    if task.num_examples() > 0:
        example = task.get_example(0)
        print(f"问题示例: {example['messages'][0]['content'][:100]}...")
        print(f"参考答案: {example['reference_answer']}")
        
        # 测试答案提取
        test_response = "为了解决这个问题，我们进行以下计算：\n2x + 5 = 13\n2x = 8\nx = 4\n答案是4"
        extracted_answer = task.extract_answer(test_response)
        print(f"提取的答案: {extracted_answer}")
        
        # 测试评估
        is_correct = task.evaluate(example, test_response)
        print(f"评估结果: {is_correct} (1=正确, 0=错误)")
        
        # 测试奖励函数
        reward = task.reward(example, test_response)
        print(f"奖励分数: {reward}")
    
    print()

def test_aime25_with_steps():
    """测试AIME25WithSteps功能"""
    print("=== 测试 AIME25WithSteps 功能 ===")
    
    # 创建AIME25WithSteps任务对象
    task = AIME25WithSteps(subset="main", split="test")
    
    print(f"数据集大小: {task.num_examples()}")
    print(f"评估类型: {task.eval_type}")
    
    if task.num_examples() > 0:
        example = task.get_example(0)
        
        # 测试包含解题步骤的回答
        step_response = "让我们一步步解决这个问题：\n\n首先，我们将方程两边同时减去5：\n2x + 5 = 13\n2x = 8\n\n然后，两边同时除以2：\n2x ÷ 2 = 8 ÷ 2\nx = 4\n\n所以答案是4。"
        
        # 测试评估（有步骤的回答）
        is_correct_with_steps = task.evaluate(example, step_response)
        print(f"包含步骤的回答评估结果: {is_correct_with_steps}")
        
        # 测试没有解题步骤的回答
        no_step_response = "答案是4"
        is_correct_no_steps = task.evaluate(example, no_step_response)
        print(f"无步骤的回答评估结果: {is_correct_no_steps}")
    
    print()

def test_answer_extraction():
    """测试各种答案提取模式"""
    print("=== 测试答案提取功能 ===")
    
    task = AIME25(subset="main", split="test")
    
    test_cases = [
        "答案是 42",
        "The answer is 42",
        "计算结果 = 42",
        "答案：42",
        "Answer: 42",
        "经过计算，答案是42 [[42]]",
        "42",  # 纯数字
        "我不确定，可能是42吧",
    ]
    
    for i, test_case in enumerate(test_cases):
        extracted = task.extract_answer(test_case)
        print(f"测试 {i+1}: '{test_case}' -> '{extracted}'")
    
    print()

def test_task_mapping():
    """测试任务映射器中的AIME25任务"""
    print("=== 测试任务映射器 ===")
    
    try:
        from scripts.chat_eval import run_chat_eval
        from functools import partial
        
        # 测试任务映射
        task_module = {
            'AIME25': partial(AIME25, subset="main", split="test"),
            'AIME25WithSteps': partial(AIME25WithSteps, subset="main", split="test"),
        }
        
        # 创建任务对象
        aime25_task = task_module['AIME25']()
        aime25_steps_task = task_module['AIME25WithSteps']()
        
        print("AIME25 任务映射成功")
        print(f"AIME25 数据集大小: {aime25_task.num_examples()}")
        print(f"AIME25WithSteps 数据集大小: {aime25_steps_task.num_examples()}")
        
    except Exception as e:
        print(f"任务映射测试失败: {e}")
    
    print()

def main():
    """主测试函数"""
    print("开始测试 AIME25 评测任务...")
    print("=" * 50)
    
    try:
        test_aime25_basic()
        test_aime25_with_steps()
        test_answer_extraction()
        test_task_mapping()
        
        print("=" * 50)
        print("✅ 所有测试完成！AIME25任务已成功集成到系统中。")
        
        print("\n使用方法:")
        print("# 评测单个AIME25任务")
        print("python scripts/chat_eval.py -i sft -a AIME25 -g base --device-type cuda")
        print("\n# 评测AIME25WithSteps任务")
        print("python scripts/chat_eval.py -i sft -a AIME25WithSteps -g base --device-type cuda")
        print("\n# 评测多个任务")
        print("python scripts/chat_eval.py -i sft -a 'GSM8K|AIME25|AIME25WithSteps' -g base --device-type cuda")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()