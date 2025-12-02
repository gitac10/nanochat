# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## 库改动说明

本项目对原始nanochat库进行了以下修改：

### AIME任务评估修改 (tasks/aime.py)
- **AIME数据集说明**：
  - AIME数据集为AIME数学竞赛乐行数据集，包含2024年和2025年的数学竞赛题目。
  - AIME数据集采用题目problem和答案answer配对的形式给出，且答案都为单个整数。
  - 由于AIME数据集答案直接为一个整数，我们选用的指标为准确率，直接比较模型给出的答案与参考答案是否完全匹配。
- **评估指标**：
  - 准确率：模型输出的答案与参考答案完全匹配的比例。
- **评估流程**：
  - 直接对模型输出的答案与参考答案进行数值比较
  - 利用Open AI 调用API接口，参考opencompass库，使用模型对答案进行评判

### 加载本地模型检查点
- **改动说明**：
  - 由于原有库代码只支持先训练后评测，修改了模型加载逻辑，新增了自定义检查点路径参数。
  - 加载了nanochat d20模型，下载地址：[nanochat_d20](https://www.modelscope.cn/models/burtenshaw/nanochat-d20/files)

### 使用方式及参数说明
- **使用方式**:
  - 直接运行python -m scripts.chat_eval,默认使用本地评估
- **chat_eval参数说明**:
  - --use_llm: 是否使用Open AI 调用API接口进行评估，默认False
  - --custom_checkpoint: 自定义模型检查点路径
  - --task-name: 任务名称，默认AIME
  - --dataset-type: 数据集类型，在使用AIME任务时，指定aime24或aime25
  
### 评估结果
- AIME 2024 数据集评估结果：
  - 准确率：0%
- AIME 2025 数据集评估结果：
  - 准确率：0%

### 注意事项
- 为了调用API接口进行评估，需要先在各类平台注册账号并获取API密钥。
- 为了使用自定义模型检查点，需要先在huggingface或魔搭等社区下载模型并保存检查点。

### 参考
- 原始nanochat库：[nanochat](https://github.com/eurekalabs/nanochat)
- opencompass库：[opencompass](https://github.com/open-compass/opencompass)
- AIME 2024 数据集：[AIME 2024](https://www.aime2024.org/)
- AIME 2025 数据集：[AIME 2025](https://www.aime2025.org/)
