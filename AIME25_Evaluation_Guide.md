# AIME25 评测任务使用指南

## 概述

AIME25 (American Invitational Mathematics Examination 2025) 是一个专为数学竞赛问题设计的评测任务。该任务包含需要精确数值答案的代数、几何、数论、组合数学等问题。

## 新增功能

### 1. AIME25 任务类
- **位置**: `tasks/aime25.py`
- **类型**: 生成式评测任务
- **特点**: 
  - 支持整数答案提取和验证
  - 包含多种答案解析模式
  - 支持强化学习奖励函数
  - 包含示例数据（实际使用时应替换为真实数据集）

### 2. AIME25WithSteps 任务类
- **位置**: `tasks/aime25.py`
- **类型**: 生成式评测任务的增强版本
- **特点**: 
  - 不仅评估答案正确性，还检查解题步骤
  - 提供更细粒度的评估

## 集成状态

AIME25任务已完全集成到nanochat评测系统中：

### ✅ 已完成的集成
1. **任务导入**: 在 `scripts/chat_eval.py` 中添加了AIME25和AIME25WithSteps的导入
2. **任务映射器**: 将新任务添加到任务映射器中
3. **默认任务列表**: 添加到所有任务的默认列表中
4. **基线准确率**: 设置为0%（开放性数学问题）

### 📝 集成代码更改

#### scripts/chat_eval.py 修改内容：

```python
# 1. 添加导入
from tasks.aime25 import AIME25, AIME25WithSteps

# 2. 添加到任务映射器
task_module = {
    'HumanEval': HumanEval,
    'MMLU': partial(MMLU, subset="all", split="test"),
    'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
    'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
    'GSM8K': partial(GSM8K, subset="main", split="test"),
    'SpellingBee': partial(SpellingBee, size=256, split="test"),
    'AIME25': partial(AIME25, subset="main", split="test"),        # 新增
    'AIME25WithSteps': partial(AIME25WithSteps, subset="main", split="test"),  # 新增
}[task_name]

# 3. 添加到默认任务列表
all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee', 'AIME25', 'AIME25WithSteps']

# 4. 设置基线准确率
baseline_accuracies = {
    'ARC-Easy': 0.25,
    'ARC-Challenge': 0.25,
    'MMLU': 0.25,
    'GSM8K': 0.0,
    'HumanEval': 0.0,
    'SpellingBee': 0.0,
    'AIME25': 0.0,              # 新增
    'AIME25WithSteps': 0.0,     # 新增
}
```

## 使用方法

### 基础用法

```bash
# 1. 评测单个AIME25任务
python scripts/chat_eval.py -i sft -a AIME25 -g base --device-type cuda

# 2. 评测AIME25WithSteps任务（包含解题步骤评估）
python scripts/chat_eval.py -i sft -a AIME25WithSteps -g base --device-type cuda

# 3. 评测多个任务（用 | 分隔）
python scripts/chat_eval.py -i sft -a "GSM8K|AIME25|AIME25WithSteps" -g base --device-type cuda

# 4. 评测所有任务（包括新的AIME25任务）
python scripts/chat_eval.py -i sft -g base --device-type cuda
```

### 高级参数

```bash
# 使用多样本评估（提高评估稳定性）
python scripts/chat_eval.py -i sft -a AIME25 -g base --device-type cuda -n 5 -t 0.1

# 增加最大生成token数（适合复杂数学问题）
python scripts/chat_eval.py -i sft -a AIME25 -g base --device-type cuda -m 1024

# 限制评估问题数量（快速测试）
python scripts/chat_eval.py -i sft -a AIME25 -g base --device-type cuda -x 50
```

### 分布式评估

```bash
# 使用8个GPU进程进行分布式评估
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a AIME25 -g base --device-type cuda
```

## 任务特点

### AIME25 标准版本
- **评估类型**: 生成式
- **答案格式**: 整数 (0-999)
- **解析方法**: 
  - 正则表达式匹配答案标签
  - 提取最后一个数字
  - 支持框选答案格式 `[[answer]]`
- **评估标准**: 精确数值匹配

### AIME25WithSteps 增强版本
- **额外功能**: 检查解题步骤的存在
- **关键词检测**: 包含数学运算关键词（如"加"、"减"、"计算"等）
- **评估标准**: 答案正确性 + 解题步骤合理性

## 示例数据

当前实现包含5个示例数学问题：

1. 基础代数方程：2x + 5 = 13
2. 几何计算：矩形面积和边长关系
3. 算术序列：前10个正整数的和
4. 函数计算：f(x) = 3x + 7
5. 指数运算：2^5 + 2^3

## 数据集扩展

### 替换为真实数据集
要使用真实的AIME25数据集，需要修改 `tasks/aime25.py` 中的 `_create_sample_data` 方法：

```python
def _create_sample_data(self, split):
    # 加载真实的AIME25数据集
    # 例如：
    # self.ds = load_dataset("your_aime25_dataset", split=split).shuffle(seed=42)
    # 返回包含以下字段的字典：
    # {
    #     "id": "唯一标识符",
    #     "problem": "数学问题文本",
    #     "answer": "正确答案",
    #     "solution": "解题步骤（可选）"
    # }
```

### 数据格式要求
- **id**: 字符串，唯一标识符
- **problem**: 字符串，数学问题的完整描述
- **answer**: 字符串或整数，正确答案
- **solution**: 字符串，解题步骤（可选）

## 注意事项

1. **数据集来源**: 当前使用的是示例数据，需要根据实际需求替换为真实的AIME25数据集
2. **答案提取**: AIME答案通常是整数，系统会自动处理数值比较
3. **性能优化**: 复杂数学问题可能需要增加 `max_new_tokens` 参数
4. **分布式评估**: 建议使用分布式评估以提高评估速度

## 与其他任务的关系

AIME25任务与现有任务的关系：
- **与GSM8K类似**: 都是数学问题，但AIME25更专注于竞赛级别的数学题目
- **与MMLU/ARC不同**: AIME25是生成式任务，不是多选题
- **与HumanEval不同**: AIME25专注于数学推理，不涉及代码生成

## ChatCORE指标

AIME25任务已包含在ChatCORE综合评估指标中。当评估所有任务时，AIME25的结果会与其他任务一起计算ChatCORE分数。