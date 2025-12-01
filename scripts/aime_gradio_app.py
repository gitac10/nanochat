#!/usr/bin/env python3
"""
AIME25 评估可视化前端界面
基于Gradio的交互式AIME数学竞赛评估工具

功能特性：
- 实时加载AIME25数据集
- 交互式数学问题解答
- 模型推理和评估
- 结果可视化图表
- 详细的答题过程分析
"""

import gradio as gr
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch

# 导入nanochat相关模块
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model, load_custom_model
from tasks.aime import AIME, AIME25WithSteps
from tasks.common import Task
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AIMEGradioApp:
    """AIME评估Gradio应用主类"""
    
    def __init__(self):
        self.aime_task = None
        self.aime_with_steps_task = None
        self.engine = None
        self.model_loaded = False
        self.evaluation_history = []
        
    def initialize_models(self, model_source, custom_checkpoint_path=None):
        """初始化模型和AIME任务"""
        try:
            # 初始化AIME任务
            self.aime_task = AIME(subset="main", split="test")
            self.aime_with_steps_task = AIME25WithSteps(subset="main", split="test")
            
            # 加载模型
            if custom_checkpoint_path:
                print(f"Loading model from custom checkpoint: {custom_checkpoint_path}")
                self.engine = load_custom_model(custom_checkpoint_path, "cuda", phase="eval")
            else:
                print(f"Loading model from source: {model_source}")
                self.engine = load_model(model_source, "cuda", phase="eval")
            
            self.model_loaded = True
            return True, f"模型加载成功！已加载 {model_source}"
            
        except Exception as e:
            self.model_loaded = False
            return False, f"模型加载失败：{str(e)}"
    
    def load_aime_problem(self, problem_id=None):
        """加载AIME问题"""
        if not self.aime_task:
            return None, None, None, None, "请先加载模型"
        
        if problem_id is None:
            # 随机选择一个问题
            problem_id = random.randint(0, len(self.aime_task.ds) - 1)
        
        example = self.aime_task.get_example(problem_id)
        problem_text = example['messages'][0]['content']
        reference_answer = example['reference_answer']
        problem_id_str = example['question_id']
        
        return problem_id_str, problem_text, reference_answer, example, ""
    
    def solve_problem(self, problem_text, use_steps_variant=False):
        """使用模型解答数学问题"""
        if not self.model_loaded or not self.engine:
            return "请先加载模型", "", 0
        
        try:
            # 准备输入
            tokenizer = self.engine.tokenizer
            bos_token_id = tokenizer.get_bos_token_id()
            
            # 编码输入
            if isinstance(problem_text, str):
                prompt_tokens = tokenizer.encode(problem_text, prepend=bos_token_id)
            else:
                prompt_tokens = problem_text
            
            # 模型推理
            start_time = time.time()
            
            # 使用batch推理API
            result = self.engine.batch_inference_api(
                inputs=[problem_text],
                batch_size=1,
                max_tokens=256,
                temperature=0.1,
                seed=42
            )
            
            inference_time = time.time() - start_time
            generated_text = result['texts'][0] if result['texts'] else ""
            
            # 提取答案
            if use_steps_variant:
                extracted_answer = self.aime_with_steps_task.extract_answer(generated_text)
                is_correct = self.aime_with_steps_task.evaluate(
                    {'reference_answer': '0'}, extracted_answer  # 临时答案用于测试
                )
            else:
                extracted_answer = self.aime_task.extract_answer(generated_text)
                is_correct = 0  # 暂时设为0，实际评估需要真实答案
            
            return generated_text, extracted_answer, is_correct
            
        except Exception as e:
            return f"推理过程中出现错误：{str(e)}", "", 0
    
    def evaluate_answer(self, problem_id, user_answer, generated_text):
        """评估用户答案"""
        if not self.aime_task:
            return "请先加载AIME任务", 0, ""
        
        try:
            # 重新获取问题数据
            example = self.aime_task.get_example(int(problem_id.split('_')[-1]))
            true_answer = example['reference_answer']
            
            # 提取模型答案
            model_answer = self.aime_task.extract_answer(generated_text)
            
            # 评估模型答案
            model_correct = self.aime_task.evaluate(example, generated_text)
            
            # 评估用户答案
            try:
                user_num = int(user_answer)
                true_num = int(true_answer)
                user_correct = 1 if user_num == true_num else 0
            except ValueError:
                user_correct = 1 if user_answer.strip() == true_answer.strip() else 0
            
            # 记录评估历史
            evaluation_record = {
                'timestamp': datetime.now().isoformat(),
                'problem_id': problem_id,
                'true_answer': true_answer,
                'user_answer': user_answer,
                'model_answer': model_answer,
                'user_correct': user_correct,
                'model_correct': model_correct,
                'model_response': generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
            }
            self.evaluation_history.append(evaluation_record)
            
            # 创建反馈
            feedback = f"""
            正确答案：{true_answer}
            
            模型答案：{model_answer} {'✅' if model_correct else '❌'}
            您的答案：{user_answer} {'✅' if user_correct else '❌'}
            
            详细分析：
            - 模型推理结果：{generated_text[:300]}{'...' if len(generated_text) > 300 else ''}
            """
            
            return feedback, user_correct, model_correct
            
        except Exception as e:
            return f"评估过程中出现错误：{str(e)}", 0, 0
    
    def generate_evaluation_chart(self):
        """生成评估结果图表"""
        if not self.evaluation_history:
            return None, "暂无评估数据"
        
        # 准备数据
        df = pd.DataFrame(self.evaluation_history)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率统计
        user_accuracy = df['user_correct'].mean() * 100
        model_accuracy = df['model_correct'].mean() * 100
        
        ax1.bar(['用户', '模型'], [user_accuracy, model_accuracy], 
                color=['#2E86C1', '#28B463'])
        ax1.set_title('答题准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for i, v in enumerate([user_accuracy, model_accuracy]):
            ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 2. 答案分布
        correct_count = df['user_correct'].sum()
        incorrect_count = len(df) - correct_count
        
        ax2.pie([correct_count, incorrect_count], 
                labels=['正确', '错误'], 
                colors=['#28B463', '#E74C3C'],
                autopct='%1.1f%%')
        ax2.set_title('用户答题分布', fontsize=14, fontweight='bold')
        
        # 3. 时间趋势
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        ax3.plot(df_sorted.index, df_sorted['user_correct'].cumsum(), 
                marker='o', linewidth=2, markersize=4)
        ax3.set_title('累计正确答题数', fontsize=14, fontweight='bold')
        ax3.set_xlabel('题目序号')
        ax3.set_ylabel('累计正确数')
        ax3.grid(True, alpha=0.3)
        
        # 4. 答案长度分析
        df['answer_length'] = df['user_answer'].astype(str).apply(len)
        ax4.hist(df['answer_length'], bins=10, color='#9B59B6', alpha=0.7, edgecolor='black')
        ax4.set_title('答案长度分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('答案长度（字符数）')
        ax4.set_ylabel('频次')
        
        plt.tight_layout()
        return fig, f"共评估了 {len(df)} 道题目"
    
    def export_results(self):
        """导出评估结果"""
        if not self.evaluation_history:
            return "暂无评估结果可导出"
        
        # 导出JSON格式
        json_data = json.dumps(self.evaluation_history, indent=2, ensure_ascii=False)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aime_evaluation_{timestamp}.json"
        
        return json_data, filename


def create_gradio_interface():
    """创建Gradio界面"""
    app = AIMEGradioApp()
    
    # 注意：已移除自定义CSS以兼容当前Gradio版本
    with gr.Blocks() as interface:
        
        # 页面标题
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background-color: #e6f3ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                    <h1 style="color: #1f4e79; margin: 0;">
                        🧮 AIME25 数学竞赛评估平台
                    </h1>
                    <p style="color: #2d5aa0; margin: 10px 0 0 0;">
                        基于Gradio的交互式数学问题求解与评估系统
                    </p>
                </div>
                """)
        
        # 第一部分：模型加载
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h2>🤖 模型配置</h2>")
                    
                    model_source = gr.Radio(
                        choices=["base", "sft", "rl"],
                        value="base",
                        label="模型来源",
                        info="选择要使用的模型类型"
                    )
                    
                    custom_checkpoint = gr.Textbox(
                        label="自定义检查点路径（可选）",
                        placeholder="例如：F:\\nanochat_d20\\",
                        info="如果提供，将使用此路径的模型"
                    )
                    
                    load_model_btn = gr.Button("🚀 加载模型", variant="primary", size="lg")
                    model_status = gr.HTML("")
        
        # 第二部分：问题展示和求解
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h2>📝 问题求解</h2>")
                    
                    with gr.Row():
                        load_problem_btn = gr.Button("🎲 随机加载问题", variant="secondary")
                        problem_id_input = gr.Textbox(label="问题ID", placeholder="AIME_001")
                        load_specific_btn = gr.Button("加载指定问题")
                    
                    problem_display = gr.HTML("")
                    reference_answer_display = gr.HTML("")
                    
                    with gr.Accordion("💡 使用模型求解", open=False):
                        solve_btn = gr.Button("🤖 使用AI模型求解", variant="primary")
                        with gr.Row():
                            use_steps_variant = gr.Checkbox(label="使用步骤评估变体", value=False)
                        
                        generated_solution = gr.Textbox(
                            label="AI生成的解答",
                            lines=8,
                            max_lines=15
                        )
                        
                        extracted_answer = gr.Textbox(label="提取的答案", placeholder="AI提取的答案将显示在这里")
                    
                    user_answer = gr.Textbox(
                        label="您的答案",
                        placeholder="请输入您的答案..."
                    )
                    
                    evaluate_btn = gr.Button("✅ 评估答案", variant="primary")
                    evaluation_result = gr.HTML("")
        
        # 第三部分：结果可视化
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h2>📊 结果分析</h2>")
                    
                    with gr.Row():
                        generate_chart_btn = gr.Button("📈 生成统计图表", variant="secondary")
                        export_btn = gr.Button("💾 导出结果", variant="secondary")
                    
                    chart_display = gr.Plot()
                    export_info = gr.HTML("")
                    export_data = gr.JSON()
        
        # 第四部分：评估历史
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h2>📋 评估历史</h2>")
                    
                    history_display = gr.Dataframe(
                        headers=["时间", "问题ID", "正确答案", "用户答案", "用户正确", "模型答案", "模型正确"],
                        row_count=10
                    )
                    
                    clear_history_btn = gr.Button("🗑️ 清空历史记录", variant="stop")
        
        # 事件绑定
        def load_model_event(source, checkpoint):
            success, message = app.initialize_models(source, checkpoint)
            if success:
                return f'<p style="color: green; font-weight: bold;">✅ {message}</p>'
            else:
                return f'<p style="color: red; font-weight: bold;">❌ {message}</p>'
        
        def load_problem_event():
            prob_id, prob_text, ref_ans, example, error = app.load_aime_problem()
            if error:
                return f'<p style="color: red; font-weight: bold;">{error}</p>', "", "", gr.update(value=[])
            else:
                problem_html = f'''
                <div style="border-left: 4px solid #2E86C1; padding: 10px; background: #f0f8ff; margin: 10px 0;">
                    <h4>📌 问题 {prob_id}</h4>
                    <p style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; font-family: 'Courier New', monospace;">
                        {prob_text}
                    </p>
                </div>
                '''
                answer_html = f'''
                <div style="border-left: 4px solid #2E86C1; padding: 10px; background: #f0f8ff; margin: 10px 0;">
                    <strong>🎯 参考答案：</strong> <span style="color: #28B463; font-weight: bold; font-size: 1.2em;">{ref_ans}</span>
                </div>
                '''
                return problem_html, answer_html, prob_id, gr.update(value=[[example['messages'][0]['content'], ref_ans, prob_id, "", ""]])
        
        def load_specific_event(prob_id):
            try:
                prob_num = int(prob_id.split('_')[-1])
                prob_id_str, prob_text, ref_ans, example, error = app.load_aime_problem(prob_num)
                if error:
                    return f'<p style="color: red; font-weight: bold;">{error}</p>', "", "", gr.update(value=[])
                else:
                    problem_html = f'''
                    <div style="border-left: 4px solid #2E86C1; padding: 10px; background: #f0f8ff; margin: 10px 0;">
                        <h4>📌 问题 {prob_id_str}</h4>
                        <p style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; font-family: 'Courier New', monospace;">
                            {prob_text}
                        </p>
                    </div>
                    '''
                    answer_html = f'''
                    <div style="border-left: 4px solid #2E86C1; padding: 10px; background: #f0f8ff; margin: 10px 0;">
                        <strong>🎯 参考答案：</strong> <span style="color: #28B463; font-weight: bold; font-size: 1.2em;">{ref_ans}</span>
                    </div>
                    '''
                    return problem_html, answer_html, prob_id_str, gr.update(value=[[example['messages'][0]['content'], ref_ans, prob_id_str, "", ""]])
            except:
                return '<p style="color: red; font-weight: bold;">❌ 问题ID格式错误，请使用 AIME_XXX 格式</p>', "", "", gr.update(value=[])
        
        def solve_event(problem_text, use_steps):
            solution, answer, correct = app.solve_problem(problem_text, use_steps)
            if "错误" in solution:
                return solution, answer, ""
            else:
                result_html = f'''
                <p style="color: green; font-weight: bold;">
                    <strong>🤖 AI解答完成</strong><br>
                    提取答案：<span style="color: #28B463; font-weight: bold;">{answer}</span>
                    {'✅ 正确' if correct else '❌ 错误'}
                </p>
                '''
                return solution, answer, result_html
        
        def evaluate_event(prob_id, user_ans, solution):
            feedback, user_correct, model_correct = app.evaluate_answer(prob_id, user_ans, solution)
            
            if user_correct:
                result_html = f'<p style="color: green; font-weight: bold;">🎉 {feedback}</p>'
            else:
                result_html = f'<p style="color: red; font-weight: bold;">💭 {feedback}</p>'
            
            return result_html
        
        def generate_chart_event():
            fig, info = app.generate_evaluation_chart()
            if fig:
                return fig, f'<p style="color: blue; font-weight: bold;">📊 {info}</p>', app.evaluation_history
            else:
                return None, f'<p style="color: red; font-weight: bold;">❌ {info}</p>', []
        
        def export_event():
            json_data, filename = app.export_results()
            return f'<p style="color: green; font-weight: bold;">💾 结果已导出为 {filename}</p>', json_data, filename
        
        def clear_history_event():
            app.evaluation_history = []
            return []
        
        # 绑定事件
        load_model_btn.click(
            load_model_event,
            inputs=[model_source, custom_checkpoint],
            outputs=[model_status]
        )
        
        load_problem_btn.click(
            load_problem_event,
            outputs=[problem_display, reference_answer_display, problem_id_input, history_display]
        )
        
        load_specific_btn.click(
            load_specific_event,
            inputs=[problem_id_input],
            outputs=[problem_display, reference_answer_display, problem_id_input, history_display]
        )
        
        solve_btn.click(
            solve_event,
            inputs=[problem_display, use_steps_variant],
            outputs=[generated_solution, extracted_answer, evaluation_result]
        )
        
        evaluate_btn.click(
            evaluate_event,
            inputs=[problem_id_input, user_answer, generated_solution],
            outputs=[evaluation_result]
        )
        
        generate_chart_btn.click(
            generate_chart_event,
            outputs=[chart_display, export_info, history_display]
        )
        
        export_btn.click(
            export_event,
            outputs=[export_info, export_data, export_info]
        )
        
        clear_history_btn.click(
            clear_history_event,
            outputs=[history_display]
        )
    
    return interface


def main():
    """主函数"""
    print("🚀 启动AIME25 Gradio评估应用...")
    print("📋 功能特性：")
    print("   - 实时加载AIME25数学竞赛问题")
    print("   - AI模型自动解题")
    print("   - 用户答案评估")
    print("   - 结果可视化分析")
    print("   - 评估历史记录")
    print("-" * 50)
    
    # 创建界面
    interface = create_gradio_interface()
    
    # 启动服务
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=False,
            show_error=True,
            quiet=True,
            debug=False
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 可能的解决方案：")
        print("   1. 检查端口7860是否被其他程序占用")
        print("   2. 检查防火墙设置是否阻止了localhost连接")
        print("   3. 尝试使用不同的端口号")
        print("   4. 检查网络代理设置")
        return


if __name__ == "__main__":
    main()