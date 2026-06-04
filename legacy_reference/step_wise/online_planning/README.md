# Online Planning Evaluation v2

一个模块化、易于调试和维护的在线规划评估系统。

## 项目结构

```
online_planning_v2/
├── main.py                          # 主入口文件
├── README.md                        # 项目文档
├── requirements.txt                 # 依赖列表
├── src/                            # 源代码目录
│   ├── __init__.py
│   ├── models/                     # 数据模型
│   │   ├── __init__.py
│   │   └── data_models.py          # 预测结果、评估结果等数据类
│   ├── clients/                    # API客户端
│   │   ├── __init__.py
│   │   └── proxy_client.py         # OpenAI兼容的代理客户端
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   └── dataset_loader.py       # 数据集加载工具
│   ├── prompts/                    # 评估提示词
│   │   ├── __init__.py
│   │   └── evaluation_prompts.py   # 不同数据集的评估提示词
│   └── evaluators/                 # 评估器
│       ├── __init__.py
│       └── online_planning_evaluator.py  # 在线规划评估器
└── config/                         # 配置文件(可选)
    └── settings.py
```

## 核心模块说明

### 1. models/data_models.py
定义了所有数据类：
- `PredictionResult`: Next-1预测结果
- `PredictionResultNext2`: Next-2预测结果  
- `EvalResult`: Next-1评估结果
- `EvalResultNext2`: Next-2评估结果
- `Statistics`: 统计信息

### 2. clients/proxy_client.py
提供与AI模型交互的客户端：
- 支持OpenAI兼容的代理API
- 支持多种模型提供商（OpenAI, Claude, Gemini, Grok）
- 支持自定义MAAS模型
- 支持图片和文本文件上传
- 支持流式和非流式响应

### 3. utils/dataset_loader.py
数据集加载和预处理：
- `DatasetLoader`: 统一加载各种benchmark数据集
- `get_base_dataset_type()`: 获取数据集的基础类型
- 支持FrameThinker格式转换

### 4. prompts/evaluation_prompts.py
评估提示词管理：
- 为不同数据集提供特定的错误分类框架
- 6种错误类型（E1-E6）的详细定义
- 评分标准（0.0-1.0）
- Next-1和Next-2评估提示词构建

### 5. evaluators/online_planning_evaluator.py
主评估器类：
- 推理（inference）：生成预测
- 评估（evaluation）：评估预测质量
- 流水线（pipeline）：推理+评估
- 支持Next-1和Next-2模式
- 支持断点续传
- 自动保存和加载进度

## 使用方法

### 环境配置

```bash
# 设置环境变量
export OPENAI_PROXY_API_KEY=${APB_API_KEY:-YOUR_API_KEY}
export GEMINI_PROXY_API_KEY=${APB_API_KEY:-YOUR_API_KEY}
export CLAUDE_PROXY_API_KEY=${APB_API_KEY:-YOUR_API_KEY}
export GROK_PROXY_API_KEY=${APB_API_KEY:-YOUR_API_KEY}
export PROXY_BASE_URL="your_base_url"
```

### 基本用法

```bash
# 1. 仅推理模式
python main.py --mode inference --dataset gta --max-samples 10

# 2. 仅评估模式
python main.py --mode evaluation --prediction-file predictions/gta_predictions_20251105_120000.json

# 3. 流水线模式（推理+评估）
python main.py --mode pipeline --dataset gta --max-samples 10

# 4. Next-2模式
python main.py --mode inference_next2 --dataset gta --max-samples 10
python main.py --mode evaluation_next2 --prediction-file predictions/next2/gta_predictions_next2.json
python main.py --mode pipeline_next2 --dataset gta --max-samples 10

# 5. 禁用断点续传
python main.py --mode pipeline --dataset gta --no-resume

# 6. 自定义模型
python main.py --mode pipeline --dataset gta \
    --test-model "gpt-4o" \
    --eval-model "claude-sonnet-4-20250514"
```

### 支持的数据集

- framethinker
- gaia
- gta  
- opencua
- skywork (及其变体: skywork_doc, skywork_excel, skywork_normal, skywork_train)
- toolbench

## 模块化优势

### 1. 易于调试
- 每个模块职责单一，便于定位问题
- 可以独立测试每个组件
- 清晰的模块边界

### 2. 易于修改
- 修改数据模型：只需修改 `models/data_models.py`
- 修改API客户端：只需修改 `clients/proxy_client.py`
- 添加新数据集：在 `utils/dataset_loader.py` 中扩展
- 修改评估标准：在 `prompts/evaluation_prompts.py` 中调整

### 3. 易于扩展
- 添加新的评估器：在 `evaluators/` 中创建新类
- 添加新的工具函数：在 `utils/` 中添加新模块
- 添加新的提示词模板：在 `prompts/` 中扩展

### 4. 代码复用
- 通过导入模块可以在其他项目中复用组件
- 每个模块都可以独立使用

## 常见调试场景

### 1. 调试数据加载
```python
from src.utils.dataset_loader import DatasetLoader

loader = DatasetLoader()
data = loader.load_dataset("gta")
print(f"Loaded {len(data)} samples")
```

### 2. 调试API客户端
```python
from src.clients.proxy_client import ProxyClient

client = ProxyClient(model="claude-sonnet-4-20250514")
response = client.generate("Hello, how are you?")
print(response)
```

### 3. 调试评估提示词
```python
from src.prompts.evaluation_prompts import EvaluationPrompts

prompt = EvaluationPrompts.build_evaluation_prompt(
    dataset_name="gta",
    query="Your query",
    trajectory_prefix="Previous steps...",
    predicted_step="Predicted next step",
    reference_next_step="Reference next step",
    reference_remaining_steps="[]",
    tools=[]
)
print(prompt)
```

### 4. 单独运行评估器的某个方法
```python
from src.evaluators.online_planning_evaluator import OnlinePlanningEvaluator

evaluator = OnlinePlanningEvaluator(
    dataset_name="gta",
    test_model="claude-sonnet-4-20250514"
)

# 只运行推理
predictions = evaluator.run_inference()
```

## 与原版对比

### 原版 (online_planning_v2.py)
- 单文件 3789 行
- 难以定位问题
- 修改风险高
- 复用困难

### 重构版 (online_planning_v2/)
- 模块化设计
- 清晰的职责分离
- 易于调试和测试
- 便于维护和扩展

## 输出目录结构

```
eval_results/
├── predictions/
│   └── {test_model}/
│       ├── next1/
│       │   └── {dataset}_predictions_{timestamp}.json
│       └── next2/
│           └── {dataset}_predictions_next2_{timestamp}.json
├── evaluations/
│   └── {test_model}/
│       ├── next1/
│       │   └── {dataset}_results_{timestamp}.json
│       └── next2/
│           └── {dataset}_results_next2_{timestamp}.json
└── resume/
    └── {test_model}/
        ├── next1/
        │   ├── {dataset}_predictions_resume.json
        │   └── {dataset}_results_resume.json
        └── next2/
            ├── {dataset}_predictions_resume.json
            └── {dataset}_results_resume.json
```

## 未来改进方向

1. 添加配置文件支持（YAML/JSON）
2. 添加单元测试
3. 添加日志系统
4. 支持并行评估
5. 添加可视化界面
6. 支持更多数据集格式

## 贡献指南

欢迎贡献代码！请遵循以下原则：
1. 保持模块化设计
2. 添加适当的文档字符串
3. 遵循现有代码风格
4. 提交前进行测试

## 许可证

请参考项目根目录的LICENSE文件。
