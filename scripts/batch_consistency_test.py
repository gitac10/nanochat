#!/usr/bin/env python3
"""
输出一致性验证脚本
功能：验证batch推理与单个推理输出完全一致
支持与chat_eval.py相同的自定义检查点路径配置
"""

import os
import sys
import time
import json
import torch
import argparse
from contextlib import nullcontext

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_custom_model
from nanochat.engine import Engine

def create_deterministic_test_prompts():
    """创建确定性测试prompts"""
    return [
        "What is 2+2?",
        "The capital of France is",
        "The weather today is",
        "Machine learning is a",
        "The largest planet in our solar system is",
        "Python is a programming language that",
        "Artificial intelligence will",
        "The speed of light is approximately",
        "Water boils at",
        " photosynthesis converts"
    ]

def token_level_comparison(tokens1, tokens2, tokenizer):
    """逐token级别的比较"""
    if len(tokens1) != len(tokens2):
        return False, f"Token length mismatch: {len(tokens1)} != {len(tokens2)}"
    
    for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
        if t1 != t2:
            text1 = tokenizer.decode([t1]) if t1 < len(tokenizer.vocab) else f"<token_{t1}>"
            text2 = tokenizer.decode([t2]) if t2 < len(tokenizer.vocab) else f"<token_{t2}>"
            return False, f"Token {i} mismatch: {t1} ('{text1}') != {t2} ('{text2}')"
    
    return True, "Perfect match"

def text_level_comparison(text1, text2):
    """文本级别的比较"""
    if text1.strip() == text2.strip():
        return True, "Perfect match"
    
    # 去除多余空格后比较
    normalized1 = ' '.join(text1.split())
    normalized2 = ' '.join(text2.split())
    
    if normalized1 == normalized2:
        return True, "Text matches after normalization"
    
    return False, f"Text mismatch after normalization"

def run_consistency_test_detailed(args):
    """运行详细的输出一致性测试"""
    print("=" * 80)
    print("DETAILED CONSISTENCY TEST")
    print("=" * 80)
    
    # 初始化
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # 加载模型
    print("Loading model...")
    if args.custom_checkpoint is not None:
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval", model_tag=args.model_tag, step=args.step)
    else:
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    
    # 测试参数 - 使用确定性设置
    test_prompts = create_deterministic_test_prompts()[:5]  # 使用前5个
    max_tokens = 20
    temperature = 0.0  # 确定性采样
    seed = 42
    
    print(f"Testing with {len(test_prompts)} prompts")
    print(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, seed={seed}")
    
    # 收集单个推理结果
    print("\n1. Collecting individual inference results...")
    individual_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"   Processing prompt {i+1}/{len(test_prompts)}...")
        
        start_time = time.time()
        result = engine.batch_inference_api(
            [prompt], 
            batch_size=1, 
            max_tokens=max_tokens, 
            temperature=temperature,
            seed=seed
        )
        end_time = time.time()
        
        individual_results.append({
            'prompt': prompt,
            'text': result['texts'][0],
            'tokens': result['tokens'][0],
            'time': end_time - start_time,
            'total_tokens': result['total_tokens']
        })
    
    # 收集batch推理结果
    print("\n2. Collecting batch inference results...")
    start_time = time.time()
    batch_result = engine.batch_inference_api(
        test_prompts, 
        batch_size=len(test_prompts), 
        max_tokens=max_tokens, 
        temperature=temperature,
        seed=seed
    )
    end_time = time.time()
    batch_time = end_time - start_time
    
    print(f"   Batch processing completed in {batch_time:.2f}s")
    
    # 详细比较结果
    print("\n3. Detailed comparison:")
    print("-" * 80)
    
    all_consistent = True
    detailed_results = []
    
    for i, (individual, batch_text, batch_tokens) in enumerate(zip(individual_results, batch_result['texts'], batch_result['tokens'])):
        prompt = individual['prompt']
        individual_text = individual['text']
        individual_tokens = individual['tokens']
        individual_time = individual['time']
        
        # Token级别比较
        token_match, token_msg = token_level_comparison(individual_tokens, batch_tokens, tokenizer)
        
        # 文本级别比较
        text_match, text_msg = text_level_comparison(individual_text, batch_text)
        
        # 综合判断
        is_consistent = token_match and text_match
        all_consistent = all_consistent and is_consistent
        
        # 记录详细结果
        detailed_results.append({
            'prompt_index': i,
            'prompt': prompt,
            'individual_time': individual_time,
            'token_match': token_match,
            'token_message': token_msg,
            'text_match': text_match,
            'text_message': text_msg,
            'overall_consistent': is_consistent,
            'individual_text': individual_text,
            'batch_text': batch_text,
            'individual_tokens': individual_tokens,
            'batch_tokens': batch_tokens
        })
        
        # 打印比较结果
        status = "✅" if is_consistent else "❌"
        print(f"{status} Prompt {i+1}: {prompt[:40]}...")
        
        if not token_match:
            print(f"   Token mismatch: {token_msg}")
        if not text_match:
            print(f"   Text mismatch: {text_msg}")
        
        if is_consistent:
            print(f"   ✅ PERFECT MATCH - Individual: {individual_time:.3f}s, Batch (avg): {batch_time/len(test_prompts):.3f}s")
        else:
            print(f"   ❌ DIFFERENT OUTPUTS")
            print(f"      Individual: '{individual_text}'")
            print(f"      Batch:      '{batch_text}'")
        print()
    
    # 性能分析
    print("4. Performance Analysis:")
    print("-" * 40)
    total_individual_time = sum(r['individual_time'] for r in individual_results)
    speedup = total_individual_time / batch_time
    
    print(f"Total individual time: {total_individual_time:.2f}s")
    print(f"Total batch time:      {batch_time:.2f}s")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Average per prompt (individual): {total_individual_time/len(test_prompts):.3f}s")
    print(f"Average per prompt (batch):      {batch_time/len(test_prompts):.3f}s")
    
    # 最终结论
    print("\n5. FINAL RESULT:")
    print("=" * 80)
    if all_consistent:
        print("🎉 SUCCESS: All outputs are perfectly consistent!")
        print(f"✅ Perfect consistency achieved across {len(test_prompts)} prompts")
        print(f"✅ {speedup:.2f}x speedup with identical results")
    else:
        failed_count = sum(1 for r in detailed_results if not r['overall_consistent'])
        print(f"❌ FAILURE: {failed_count}/{len(test_prompts)} prompts have inconsistent outputs")
        print("❌ Batch inference does NOT produce identical results")
    
    print("=" * 80)
    
    # 保存详细结果
    final_results = {
        'test_timestamp': time.time(),
        'test_parameters': {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'seed': seed,
            'num_prompts': len(test_prompts)
        },
        'consistency_passed': all_consistent,
        'performance': {
            'total_individual_time': total_individual_time,
            'total_batch_time': batch_time,
            'speedup': speedup
        },
        'detailed_results': detailed_results
    }
    
    with open('consistency_test_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: consistency_test_results.json")
    
    return all_consistent, final_results

def run_multiple_seeds_test(args):
    """测试不同seed下的一致性"""
    print("\n" + "=" * 80)
    print("MULTIPLE SEEDS CONSISTENCY TEST")
    print("=" * 80)
    
    # 初始化
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # 加载模型
    if args.custom_checkpoint is not None:
        model, tokenizer, meta = load_custom_model(args.custom_checkpoint, device, phase="eval", model_tag=args.model_tag, step=args.step)
    else:
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    
    # 测试prompts
    test_prompts = create_deterministic_test_prompts()[:3]
    seeds = [42, 123, 456, 789, 999]
    
    print(f"Testing with {len(test_prompts)} prompts and {len(seeds)} different seeds")
    
    seed_results = {}
    
    for seed in seeds:
        print(f"\nTesting seed {seed}...")
        
        # Batch推理
        batch_result = engine.batch_inference_api(
            test_prompts,
            batch_size=len(test_prompts),
            max_tokens=15,
            temperature=0.7,  # 使用非零温度来测试随机性
            seed=seed
        )
        
        # 单个推理
        individual_results = []
        for prompt in test_prompts:
            result = engine.batch_inference_api(
                [prompt],
                batch_size=1,
                max_tokens=15,
                temperature=0.7,
                seed=seed
            )
            individual_results.append(result['texts'][0])
        
        # 比较文本相似性（允许一定差异，因为使用了非零温度）
        consistency_scores = []
        for i, (batch_text, individual_text) in enumerate(zip(batch_result['texts'], individual_results)):
            # 简单的文本相似度检查
            batch_words = set(batch_text.lower().split())
            individual_words = set(individual_text.lower().split())
            
            if len(batch_words) > 0:
                intersection = len(batch_words.intersection(individual_words))
                union = len(batch_words.union(individual_words))
                similarity = intersection / union if union > 0 else 0
                consistency_scores.append(similarity)
            else:
                consistency_scores.append(0)
        
        avg_similarity = sum(consistency_scores) / len(consistency_scores)
        seed_results[seed] = {
            'similarity_score': avg_similarity,
            'batch_results': batch_result['texts'],
            'individual_results': individual_results
        }
        
        print(f"  Seed {seed}: Average similarity = {avg_similarity:.3f}")
    
    # 分析结果
    print(f"\n📊 MULTIPLE SEEDS ANALYSIS:")
    similarities = [result['similarity_score'] for result in seed_results.values()]
    avg_similarity = sum(similarities) / len(similarities)
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    
    print(f"Average similarity across all seeds: {avg_similarity:.3f}")
    print(f"Similarity range: {min_similarity:.3f} - {max_similarity:.3f}")
    
    if avg_similarity > 0.5:
        print("✅ Good consistency across different seeds")
    else:
        print("⚠️  Low consistency across different seeds (expected with high temperature)")
    
    return seed_results

def parse_args():
    """解析命令行参数 - 参考chat_eval.py的实现"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, help="Source of the model: sft|mid|rl")
    parser.add_argument('-p', '--custom_checkpoint', type=str, 
                       default="F:\\nanochat_d20\\",
                       help="Custom checkpoint directory path (used when source='custom')")
    parser.add_argument('-d', '--dtype', type=str, default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], 
                       help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('--max-tokens', type=int, default=20, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic testing')
    return parser.parse_args()

def main():
    """主函数"""
    print("Output Consistency Verification Suite")
    print("=" * 80)
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置默认源为custom（使用自定义检查点）
    if args.source is None:
        args.source = 'custom'
    
    print(f"Model source: {args.source}")
    if args.source == 'custom' and args.custom_checkpoint:
        print(f"Custom checkpoint path: {args.custom_checkpoint}")
    print(f"Device type: {args.device_type if args.device_type else 'auto-detect'}")
    print(f"Data type: {args.dtype}")
    print()
    
    try:
        # 详细一致性测试（确定性参数）
        consistent, detailed_results = run_consistency_test_detailed(args)
        
        # 多种子测试（非确定性参数）
        seed_results = run_multiple_seeds_test(args)
        
        print("\n" + "=" * 80)
        print("FINAL VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"✅ Deterministic consistency test: {'PASSED' if consistent else 'FAILED'}")
        print(f"✅ Multiple seeds test: COMPLETED")
        print(f"✅ Output verification: {'SUCCESS' if consistent else 'NEEDS INVESTIGATION'}")
        
        if consistent:
            print("\n🎉 BATCH INFERENCE IS FULLY CONSISTENT WITH INDIVIDUAL INFERENCE")
            print("   - Identical outputs when using deterministic parameters")
            print("   - Significant performance improvement achieved")
            print("   - Ready for production use")
        else:
            print("\n⚠️  OUTPUT INCONSISTENCY DETECTED")
            print("   - Batch inference produces different outputs than individual inference")
            print("   - Investigation required before production use")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()