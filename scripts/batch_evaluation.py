#!/usr/bin/env python3
"""
Batch推理评测脚本
功能：测试批量推理的正确性和性能
支持自定义检查点路径配置（通过--custom_checkpoint/-p参数）
"""

import os
import sys
import time
import json
import torch
import argparse
from contextlib import nullcontext

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_custom_model
from nanochat.engine import Engine

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Batch推理评测脚本')
    parser.add_argument('--source', '-i', type=str, default='custom', 
                       choices=['base', 'custom'], 
                       help='模型源类型: base 或 custom')
    parser.add_argument('--custom_checkpoint', '-p', type=str, 
                       default=r'F:\\nanochat_d20\\',
                       help='自定义检查点路径 (当 source=custom 时使用)')
    parser.add_argument('--device', type=str, default=None,
                       help='指定设备类型 (auto/cpu/cuda), 默认为自动检测')
    parser.add_argument('--dtype', type=str, default=None,
                       help='数据类型 (auto/float16/bfloat16/float32), 默认为自动检测')
    
    return parser.parse_args()

def generate_test_prompts():
    """生成测试用的prompts"""
    return [
        "What is the capital of France?",
        "Explain the concept of artificial intelligence.",
        "How does photosynthesis work?",
        "What is 15 times 27?",
        "Write a short poem about spring.",
        "What are the main components of a computer?",
        "How do vaccines work?",
        "What is the meaning of life?",
        "Explain quantum physics in simple terms.",
        "What causes seasons to change?"
    ]

def test_basic_batch_inference(args):
    """测试基本的batch推理功能"""
    print("=" * 60)
    print("Basic Batch Inference Test")
    print("=" * 60)
    
    # 初始化
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type() if args.device is None else args.device
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 加载模型
    print("Loading model...")
    if args.source == "custom" and args.custom_checkpoint:
        print(f"Using custom checkpoint: {args.custom_checkpoint}")
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval")
    else:
        print(f"Using source: {args.source}")
        model, tokenizer, meta = load_model(args.source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # 生成测试数据
    prompts = generate_test_prompts()
    print(f"Testing with {len(prompts)} prompts")
    
    # 测试字符串输入
    print("\n1. Testing string inputs...")
    start_time = time.time()
    result = engine.batch_inference_api(prompts, max_tokens=32, temperature=0.7)
    end_time = time.time()
    
    print(f"Batch size: {result['batch_size']}")
    print(f"Inference time: {end_time - start_time:.2f}s")
    print(f"Throughput: {result['throughput']:.2f} prompts/second")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Tokens per second: {result['tokens_per_second']:.2f}")
    
    # 显示前3个结果
    print("\nSample outputs:")
    for i in range(min(3, len(result['texts']))):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Response: {result['texts'][i][:100]}...")
        print()
    
    return result

def test_different_batch_sizes(args):
    """测试不同batch size的性能"""
    print("=" * 60)
    print("Different Batch Size Performance Test")
    print("=" * 60)
    
    # 初始化
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type() if args.device is None else args.device
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 加载模型
    if args.source == "custom" and args.custom_checkpoint:
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval")
    else:
        model, tokenizer, meta = load_model(args.source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # 生成测试数据
    prompts = generate_test_prompts() * 3  # 生成30个prompts
    batch_sizes = [1, 2, 4, 8, 16, len(prompts)]
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(prompts):
            continue
            
        print(f"\nTesting batch size: {batch_size}")
        start_time = time.time()
        
        result = engine.batch_inference_api(
            prompts[:batch_size], 
            batch_size=batch_size, 
            max_tokens=32, 
            temperature=0.7
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results[batch_size] = {
            'inference_time': total_time,
            'throughput': result['throughput'],
            'tokens_per_second': result['tokens_per_second'],
            'total_tokens': result['total_tokens']
        }
        
        print(f"  Time: {total_time:.2f}s")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")
        print(f"  Tokens/sec: {result['tokens_per_second']:.2f}")
        print(f"  Total tokens: {result['total_tokens']}")
    
    return results

def test_consistency(args):
    """测试batch推理与单个推理的一致性"""
    print("=" * 60)
    print("Consistency Test (Batch vs Individual)")
    print("=" * 60)
    
    # 初始化
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type() if args.device is None else args.device
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 加载模型
    if args.source == "custom" and args.custom_checkpoint:
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval")
    else:
        model, tokenizer, meta = load_model(args.source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # 选择少量prompts进行测试
    prompts = generate_test_prompts()[:3]
    
    # 单个推理
    print("Running individual inferences...")
    individual_results = []
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        result = engine.batch_inference_api([prompt], batch_size=1, max_tokens=16, temperature=0.0, seed=42)
        end_time = time.time()
        
        individual_results.append({
            'text': result['texts'][0],
            'tokens': result['tokens'][0],
            'time': end_time - start_time
        })
        print(f"  Prompt {i+1}: {individual_results[-1]['text']}")
    
    # Batch推理（相同seed）
    print("\nRunning batch inference...")
    start_time = time.time()
    batch_result = engine.batch_inference_api(prompts, batch_size=len(prompts), max_tokens=16, temperature=0.0, seed=42)
    end_time = time.time()
    batch_time = end_time - start_time
    
    print(f"Batch time: {batch_time:.2f}s")
    print(f"Individual total time: {sum(r['time'] for r in individual_results):.2f}s")
    print(f"Speedup: {sum(r['time'] for r in individual_results) / batch_time:.2f}x")
    
    # 比较结果
    print("\nComparing outputs:")
    all_consistent = True
    for i, (individual, batch) in enumerate(zip(individual_results, batch_result['texts'])):
        consistent = individual['text'] == batch
        all_consistent = all_consistent and consistent
        status = "✅ CONSISTENT" if consistent else "❌ DIFFERENT"
        print(f"  Prompt {i+1}: {status}")
        if not consistent:
            print(f"    Individual: {individual['text']}")
            print(f"    Batch:      {batch}")
    
    print(f"\nOverall consistency: {'✅ PASS' if all_consistent else '❌ FAIL'}")
    return all_consistent

def test_variable_length_sequences(args):
    """测试不同长度序列的处理"""
    print("=" * 60)
    print("Variable Length Sequences Test")
    print("=" * 60)
    
    # 初始化
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type() if args.device is None else args.device
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 加载模型
    if args.source == "custom" and args.custom_checkpoint:
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval")
    else:
        model, tokenizer, meta = load_model(args.source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # 创建不同长度的prompts
    prompts = [
        "Short",  # 1 token
        "This is a medium length prompt for testing purposes.",  # 约12 tokens
        "This is a much longer prompt that should test how the engine handles sequences of varying lengths when processing them in a batch together. We want to make sure that padding and attention masks work correctly.",  # 约40 tokens
        "Another short one.",
        "Medium length test prompt with some complexity.",
    ]
    
    print(f"Testing {len(prompts)} prompts of varying lengths:")
    for i, prompt in enumerate(prompts):
        token_count = len(tokenizer.encode(prompt))
        print(f"  Prompt {i+1}: {token_count} tokens - {prompt[:50]}...")
    
    # 测试batch推理
    print("\nRunning batch inference...")
    start_time = time.time()
    result = engine.batch_inference_api(prompts, max_tokens=16, temperature=0.7)
    end_time = time.time()
    
    print(f"Batch processing completed in {end_time - start_time:.2f}s")
    print(f"Generated {len(result['texts'])} responses")
    
    print("\nGenerated responses:")
    for i, (prompt, response) in enumerate(zip(prompts, result['texts'])):
        print(f"  {i+1}. {prompt[:30]}... -> {response[:50]}...")
    
    return result

def main():
    """主测试函数"""
    args = parse_args()
    
    print("Batch Inference Evaluation Script")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Source: {args.source}")
    if args.source == "custom":
        print(f"  Custom checkpoint: {args.custom_checkpoint}")
    print(f"  Device: {args.device or 'auto-detected'}")
    print(f"  Data type: {args.dtype or 'auto-detected'}")
    print("=" * 60)
    
    try:
        # 测试基本功能
        basic_result = test_basic_batch_inference(args)
        
        # 测试不同batch size
        print("\n")
        performance_results = test_different_batch_sizes(args)
        
        # 测试一致性
        print("\n")
        consistency_result = test_consistency(args)
        
        # 测试变长序列
        print("\n")
        variable_result = test_variable_length_sequences(args)
        
        # 保存结果
        results_summary = {
            'timestamp': time.time(),
            'basic_test': {
                'throughput': basic_result['throughput'],
                'tokens_per_second': basic_result['tokens_per_second'],
                'batch_size': basic_result['batch_size']
            },
            'performance_by_batch_size': performance_results,
            'consistency_test_passed': consistency_result,
            'variable_length_test_completed': True
        }
        
        with open('batch_evaluation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"✅ Basic batch inference: SUCCESS")
        print(f"✅ Performance testing: {len(performance_results)} batch sizes tested")
        print(f"✅ Consistency test: {'PASSED' if consistency_result else 'FAILED'}")
        print(f"✅ Variable length sequences: SUCCESS")
        print(f"✅ Results saved to: batch_evaluation_results.json")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()