#!/usr/bin/env python3
"""
Batch推理性能基准测试脚本
功能：测试不同batch size下的性能提升效果
支持与chat_eval.py相同的自定义检查点路径配置
"""

import os
import sys
import time
import json
import torch
import numpy as np
import argparse
import statistics
from contextlib import nullcontext

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_custom_model
from nanochat.engine import Engine

class BatchPerformanceBenchmark:
    """Batch推理性能基准测试"""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.model = None
        self.engine = None
        self.tokenizer = None
        
    def setup(self):
        """初始化模型和引擎"""
        print("Setting up benchmark environment...")
        
        device_type = autodetect_device_type() if self.args.device_type == "" else self.args.device_type
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
        ptdtype = torch.float32 if self.args.dtype == 'float32' else torch.bfloat16
        self.autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

        print("Loading model...")
        if self.args.custom_checkpoint is not None:
            model, tokenizer, meta = load_custom_model(self.args.custom_checkpoint, device, phase="eval", model_tag=self.args.model_tag, step=self.args.step)
        else:
            model, tokenizer, meta = load_model(self.args.source, device, phase="eval", model_tag=self.args.model_tag, step=self.args.step)
        
        self.model = model
        self.tokenizer = tokenizer
        self.engine = Engine(model, tokenizer)
        self.device = device
        
        print("Setup complete!")
        
    def generate_benchmark_prompts(self, num_prompts=100):
        """生成基准测试用的prompts"""
        prompt_templates = [
            "What is the capital of {}?",
            "Explain how {} works in simple terms.",
            "What is the result of {}?",
            "List three facts about {}.",
            "How does {} differ from {}?",
            "What are the benefits of {}?",
            "Explain the process of {}.",
            "What causes {}?",
            "Describe {} in detail.",
            "Compare {} and {}."
        ]
        
        test_subjects = [
            "France", "artificial intelligence", "photosynthesis", "vaccines",
            "quantum physics", "climate change", "machine learning", "blockchain",
            "renewable energy", "DNA", "neurons", "democracy", "economics",
            "philosophy", "chemistry", "biology", "psychology", "sociology"
        ]
        
        prompts = []
        for i in range(num_prompts):
            template = prompt_templates[i % len(prompt_templates)]
            subject = test_subjects[i % len(test_subjects)]
            
            if "{}" in template and template.count("{}") == 2:
                subject2 = test_subjects[(i + 1) % len(test_subjects)]
                prompt = template.format(subject, subject2)
            elif "{}" in template:
                prompt = template.format(subject)
            else:
                prompt = template
                
            prompts.append(prompt)
            
        return prompts
    
    def measure_individual_inference(self, prompts, num_runs=3):
        """测量单个推理的性能（作为基准）"""
        print(f"\nMeasuring individual inference performance ({num_runs} runs)...")
        
        times = []
        for run in range(num_runs):
            run_times = []
            for prompt in prompts:
                start_time = time.time()
                
                result = self.engine.batch_inference_api(
                    [prompt], 
                    batch_size=1, 
                    max_tokens=32, 
                    temperature=0.7
                )
                
                end_time = time.time()
                run_times.append(end_time - start_time)
            
            total_time = sum(run_times)
            avg_time = total_time / len(prompts)
            times.append(avg_time)
            
            print(f"  Run {run + 1}: {avg_time:.2f}s total, {avg_time/len(prompts):.3f}s per prompt")
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        return {
            'total_time': avg_time,
            'time_per_prompt': avg_time / len(prompts),
            'prompts_per_second': len(prompts) / avg_time,
            'std_dev': std_time,
            'num_prompts': len(prompts)
        }
    
    def measure_batch_inference(self, prompts, batch_sizes, num_runs=3):
        """测量batch推理的性能"""
        print(f"\nMeasuring batch inference performance...")
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(prompts):
                continue
                
            print(f"  Testing batch size: {batch_size}")
            batch_times = []
            
            for run in range(num_runs):
                batch_prompts = prompts[:batch_size]
                
                start_time = time.time()
                
                result = self.engine.batch_inference_api(
                    batch_prompts, 
                    batch_size=batch_size, 
                    max_tokens=32, 
                    temperature=0.7
                )
                
                end_time = time.time()
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                
                print(f"    Run {run + 1}: {batch_time:.2f}s")
            
            avg_time = statistics.mean(batch_times)
            std_time = statistics.stdev(batch_times) if len(batch_times) > 1 else 0
            
            results[batch_size] = {
                'total_time': avg_time,
                'time_per_prompt': avg_time / batch_size,
                'prompts_per_second': batch_size / avg_time,
                'std_dev': std_time,
                'batch_size': batch_size,
                'num_prompts': batch_size
            }
            
            print(f"    Average: {avg_time:.2f}s total, {avg_time/batch_size:.3f}s per prompt, {batch_size/avg_time:.2f} prompts/sec")
        
        return results
    
    def calculate_speedup(self, individual_results, batch_results):
        """计算加速比"""
        speedup_results = {}
        
        individual_pps = individual_results['prompts_per_second']
        
        for batch_size, batch_result in batch_results.items():
            batch_pps = batch_result['prompts_per_second']
            speedup = batch_pps / individual_pps
            efficiency = speedup / batch_size * 100  # 效率百分比
            
            speedup_results[batch_size] = {
                'speedup': speedup,
                'efficiency': efficiency,
                'individual_pps': individual_pps,
                'batch_pps': batch_pps
            }
        
        return speedup_results
    
    def run_comprehensive_benchmark(self, num_prompts=50, batch_sizes=None):
        """运行综合基准测试"""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        print("=" * 70)
        print("COMPREHENSIVE BATCH INFERENCE BENCHMARK")
        print("=" * 70)
        print(f"Testing with {num_prompts} prompts")
        print(f"Batch sizes to test: {batch_sizes}")
        
        # 生成测试数据
        prompts = self.generate_benchmark_prompts(num_prompts)
        print(f"Generated {len(prompts)} test prompts")
        
        # 测量单个推理性能
        individual_results = self.measure_individual_inference(prompts[:10])  # 用前10个测试
        
        # 测量batch推理性能
        batch_results = self.measure_batch_inference(prompts, batch_sizes)
        
        # 计算加速比
        speedup_results = self.calculate_speedup(individual_results, batch_results)
        
        # 保存结果
        benchmark_results = {
            'timestamp': time.time(),
            'num_prompts_tested': num_prompts,
            'batch_sizes_tested': batch_sizes,
            'individual_baseline': individual_results,
            'batch_results': batch_results,
            'speedup_analysis': speedup_results,
            'hardware_info': {
                'device': str(self.device),
                'model_name': self.model.config.name if hasattr(self.model.config, 'name') else 'unknown'
            }
        }
        
        self.results = benchmark_results
        return benchmark_results
    
    def print_summary_table(self):
        """打印结果汇总表"""
        if not self.results:
            print("No results to summarize. Run benchmark first.")
            return
        
        results = self.results
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Batch Size':<12} {'Time (s)':<12} {'Prompts/sec':<15} {'Speedup':<10} {'Efficiency':<12}")
        print("-" * 80)
        
        # 基准结果
        baseline = results['individual_baseline']
        print(f"{'Individual':<12} {baseline['total_time']:<12.2f} {baseline['prompts_per_second']:<15.2f} {'1.00x':<10} {'100%':<12}")
        
        # Batch结果
        for batch_size in sorted(results['batch_results'].keys()):
            batch_result = results['batch_results'][batch_size]
            speedup = results['speedup_analysis'][batch_size]['speedup']
            efficiency = results['speedup_analysis'][batch_size]['efficiency']
            
            print(f"{batch_size:<12} {batch_result['total_time']:<12.2f} {batch_result['prompts_per_second']:<15.2f} {f'{speedup:.2f}x':<10} {f'{efficiency:.1f}%':<12}")
        
        print("=" * 80)
        
        # 最佳性能
        best_batch_size = max(results['speedup_analysis'].keys(), 
                             key=lambda x: results['speedup_analysis'][x]['speedup'])
        best_speedup = results['speedup_analysis'][best_batch_size]['speedup']
        
        print(f"\n🎯 BEST PERFORMANCE:")
        print(f"   Batch size {best_batch_size}: {best_speedup:.2f}x speedup")
        print(f"   Throughput: {results['batch_results'][best_batch_size]['prompts_per_second']:.2f} prompts/second")
    
    def save_detailed_results(self, filename='batch_performance_results.json'):
        """保存详细结果到JSON文件"""
        if not self.results:
            print("No results to save. Run benchmark first.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {filename}")
    
    def generate_performance_plot(self):
        """生成性能图表"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            speedup_data = self.results['speedup_analysis']
            batch_sizes = sorted(speedup_data.keys())
            speedups = [speedup_data[bs]['speedup'] for bs in batch_sizes]
            efficiencies = [speedup_data[bs]['efficiency'] for bs in batch_sizes]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 加速比图
            ax1.plot(batch_sizes, speedups, 'bo-', linewidth=2, markersize=8)
            ax1.plot(batch_sizes, batch_sizes, 'r--', alpha=0.7, label='Ideal Linear Speedup')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Speedup (x)')
            ax1.set_title('Batch Inference Speedup')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 效率图
            ax2.plot(batch_sizes, efficiencies, 'go-', linewidth=2, markersize=8)
            ax2.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100% Efficiency')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Efficiency (%)')
            ax2.set_title('Batch Inference Efficiency')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('batch_performance_plots.png', dpi=300, bbox_inches='tight')
            print("Performance plots saved to: batch_performance_plots.png")
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")

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
    parser.add_argument('--num-prompts', type=int, default=40, help='Number of prompts for testing')
    parser.add_argument('--max-tokens', type=int, default=32, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of test runs to average')
    return parser.parse_args()

def main():
    """主函数"""
    print("Batch Inference Performance Benchmark")
    print("=" * 70)
    
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
    print(f"Test prompts: {args.num_prompts}")
    print()
    
    try:
        # 创建基准测试实例
        benchmark = BatchPerformanceBenchmark(args)
        
        # 设置环境
        benchmark.setup()
        
        # 运行基准测试
        batch_sizes = [1, 2, 4, 8, 16, 32]
        results = benchmark.run_comprehensive_benchmark(num_prompts=args.num_prompts, batch_sizes=batch_sizes)
        
        # 打印汇总
        benchmark.print_summary_table()
        
        # 保存结果
        benchmark.save_detailed_results()
        benchmark.generate_performance_plot()
        
        print(f"\n✅ Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()