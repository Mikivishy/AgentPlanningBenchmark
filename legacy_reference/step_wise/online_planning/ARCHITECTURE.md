# 项目架构图

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                   (命令行入口 & 参数解析)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              OnlinePlanningEvaluator                         │
│                   (主评估器协调器)                            │
│  ┌──────────────┬──────────────┬──────────────────┐         │
│  │ run_inference│ run_evaluation│ run_pipeline     │         │
│  └──────────────┴──────────────┴──────────────────┘         │
└───┬────────┬────────┬────────┬─────────────────────────────┘
    │        │        │        │
    ▼        ▼        ▼        ▼
┌─────┐  ┌─────┐  ┌──────┐  ┌────────┐
│Models│  │Client│  │Utils │  │Prompts │
└─────┘  └─────┘  └──────┘  └────────┘
```

## 模块依赖关系

```
main.py
  └── evaluators/online_planning_evaluator.py
        ├── models/data_models.py
        │     └── (PredictionResult, EvalResult, Statistics等)
        │
        ├── clients/proxy_client.py  
        │     └── (ProxyClient调用AI API)
        │
        ├── utils/dataset_loader.py
        │     └── (DatasetLoader加载数据)
        │
        └── prompts/evaluation_prompts.py
              └── (EvaluationPrompts生成提示词)
```

## 数据流向

### 1. 推理模式（Inference）

```
User Input (CLI)
    │
    ├─→ main.py --mode inference --dataset gta
    │
    ├─→ OnlinePlanningEvaluator.__init__()
    │     ├─→ ProxyClient(test_model)
    │     └─→ DatasetLoader.load_dataset(gta)
    │
    ├─→ OnlinePlanningEvaluator.run_inference()
    │     │
    │     ├─→ 遍历每个样本
    │     │     │
    │     │     ├─→ 构建提示词
    │     │     ├─→ ProxyClient.generate()
    │     │     ├─→ 解析预测结果
    │     │     └─→ 创建PredictionResult
    │     │
    │     └─→ 返回 List[PredictionResult]
    │
    └─→ save_predictions()
          └─→ predictions/{model}/next1/predictions.json
```

### 2. 评估模式（Evaluation）

```
User Input (CLI)
    │
    ├─→ main.py --mode evaluation --prediction-file xxx.json
    │
    ├─→ OnlinePlanningEvaluator.load_predictions()
    │     └─→ 加载PredictionResult列表
    │
    ├─→ OnlinePlanningEvaluator.run_evaluation()
    │     │
    │     ├─→ 遍历每个预测
    │     │     │
    │     │     ├─→ EvaluationPrompts.build_evaluation_prompt()
    │     │     ├─→ ProxyClient.generate()
    │     │     ├─→ 解析评估结果
    │     │     └─→ 创建EvalResult
    │     │
    │     └─→ 返回 List[EvalResult]
    │
    ├─→ calculate_statistics()
    │     └─→ 生成Statistics
    │
    └─→ save_results()
          └─→ evaluations/{model}/next1/results.json
```

### 3. 流水线模式（Pipeline）

```
User Input (CLI)
    │
    ├─→ main.py --mode pipeline --dataset gta
    │
    └─→ OnlinePlanningEvaluator.run_pipeline()
          │
          ├─→ run_inference()
          │     └─→ List[PredictionResult]
          │
          ├─→ run_evaluation()
          │     └─→ List[EvalResult]
          │
          ├─→ calculate_statistics()
          │     └─→ Statistics
          │
          └─→ save_results()
```

## 类关系图

```
┌─────────────────────────────────┐
│      OnlinePlanningEvaluator     │
│  ┌────────────────────────────┐ │
│  │  - dataset_name            │ │
│  │  - test_model              │ │
│  │  - eval_model              │ │
│  │  - test_client             │ │◄─────┐
│  │  - eval_client             │ │      │
│  │  - data                    │ │◄───┐ │
│  └────────────────────────────┘ │    │ │
│  ┌────────────────────────────┐ │    │ │
│  │  + run_inference()         │ │    │ │
│  │  + run_evaluation()        │ │    │ │
│  │  + run_pipeline()          │ │    │ │
│  │  + calculate_statistics()  │ │    │ │
│  └────────────────────────────┘ │    │ │
└─────────────────────────────────┘    │ │
                                       │ │
┌─────────────────────┐    ┌──────────┴─┴───────┐
│   ProxyClient       │◄───│  DatasetLoader     │
│  ┌───────────────┐ │    │  ┌───────────────┐ │
│  │ - model       │ │    │  │ + load_dataset│ │
│  │ - client      │ │    │  └───────────────┘ │
│  └───────────────┘ │    └────────────────────┘
│  ┌───────────────┐ │
│  │ + generate()  │ │
│  └───────────────┘ │
└─────────────────────┘

┌─────────────────────────────────┐
│    EvaluationPrompts             │
│  ┌────────────────────────────┐ │
│  │ + get_error_definitions()  │ │
│  │ + get_scoring_rubric()     │ │
│  │ + build_evaluation_prompt()│ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘

┌──────────────────┐  ┌───────────────┐  ┌─────────────────┐
│ PredictionResult │  │  EvalResult   │  │  Statistics     │
│ ┌──────────────┐ │  │ ┌───────────┐ │  │ ┌─────────────┐ │
│ │ - task_id    │ │  │ │ - task_id │ │  │ │ - total     │ │
│ │ - dataset    │ │  │ │ - score   │ │  │ │ - accuracy  │ │
│ │ - query      │ │  │ │ - errors  │ │  │ │ - avg_score │ │
│ │ - predicted  │ │  │ └───────────┘ │  │ └─────────────┘ │
│ └──────────────┘ │  └───────────────┘  └─────────────────┘
└──────────────────┘
```

## 文件间通信

```
┌─────────────────────────────────────────────────────────┐
│                      文件系统                            │
│                                                          │
│  predictions/                                            │
│    {model}/next1/{dataset}_predictions.json              │
│         ▲                                                │
│         │ save_predictions()                             │
│         │                                                │
│  ┌──────┴────────────────────────┐                      │
│  │  OnlinePlanningEvaluator      │                      │
│  │    run_inference()            │                      │
│  └──────┬────────────────────────┘                      │
│         │ save_results()                                 │
│         ▼                                                │
│  evaluations/                                            │
│    {model}/next1/{dataset}_results.json                  │
│                                                          │
│  resume/                                                 │
│    {model}/next1/                                        │
│      {dataset}_predictions_resume.json                   │
│      {dataset}_results_resume.json                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 断点续传机制

```
开始执行
    │
    ├─→ 检查resume目录
    │     │
    │     ├─→ 有resume文件？
    │     │     ├─→ YES: 加载已完成的task_id集合
    │     │     └─→ NO: 空集合
    │     │
    ├─→ 遍历样本
    │     │
    │     ├─→ task_id已完成？
    │     │     ├─→ YES: 跳过
    │     │     └─→ NO: 处理
    │     │
    │     ├─→ 处理完成
    │     │
    │     └─→ 每5个样本
    │           └─→ save_resume_file()
    │
    └─→ 完成
          └─→ 删除resume文件
```

## 错误处理流程

```
API调用
    │
    ├─→ 尝试发送请求
    │     │
    │     ├─→ 成功？
    │     │     └─→ 返回结果
    │     │
    │     └─→ 失败
    │           │
    │           ├─→ 记录错误
    │           ├─→ 保存当前进度到resume
    │           └─→ 抛出异常
    │
    └─→ 用户可以从断点继续
```

## 扩展点

```
1. 添加新数据集
   └── 修改: utils/dataset_loader.py
        └── 添加数据集加载逻辑

2. 添加新模型
   └── 修改: clients/proxy_client.py
        └── 添加模型provider识别

3. 添加新评估标准
   └── 修改: prompts/evaluation_prompts.py
        └── 添加新的error definitions

4. 添加新评估器
   └── 创建: evaluators/custom_evaluator.py
        └── 继承OnlinePlanningEvaluator

5. 添加新数据模型
   └── 修改: models/data_models.py
        └── 添加新的@dataclass
```
