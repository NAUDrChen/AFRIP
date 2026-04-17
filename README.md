# AFRIP

AFRIP（Advanced Framework for Radar Information Processing）是一个面向雷达信息智能处理的模块化研发框架，目标是支持检测、跟踪、训练、评估与工具链的高内聚、低耦合组合。

## 设计目标

- 统一组织检测模型、跟踪模型与通用处理模块
- 将数据、模型、策略、运行流程和配置分层解耦
- 通过注册机制与组合式配置支持快速试验
- 便于逐步扩展到多任务、多模态与多阶段流程

## 当前骨架

```text
AFRIP/
├─ configs/                 # 配置层：基础配置、数据、模型、策略、实验组合
├─ docs/                    # 架构说明与设计文档
├─ outputs/                 # 训练日志、权重、评估结果等输出目录
├─ scripts/                 # 训练、评估、导出等脚本入口
├─ src/afrip/               # 主代码包
│  ├─ core/                 # 注册器、构建器、基础抽象
│  ├─ datasets/             # 数据集、加载器、增强、采样器
│  ├─ models/               # 检测、跟踪及模型子组件
│  ├─ modules/              # 预处理、后处理等通用模块
│  ├─ engine/               # 训练/评估/推理执行引擎
│  ├─ strategies/           # 优化器、调度器、训练策略、预训练加载策略
│  ├─ evaluation/           # 指标、评测协议、可视化分析
│  └─ utils/                # 配置、日志、随机种子等工具
└─ tests/                   # 单元测试
```

## 快速开始

1. 安装开发依赖
2. 运行示例脚本验证配置组合

```powershell
pip install -e .[dev]
python .\scripts\train.py --config .\configs\experiments\detection\radardet_rdcnn_sort.yaml
pytest
```

## 配置设计思路

- `configs/base/`：运行时、设备、日志、输出等全局基础配置
- `configs/datasets/`：数据源、切分、增强、采样与 dataloader 配置
- `configs/detectors/`：检测模型结构与超参数
- `configs/trackers/`：跟踪模型结构与超参数
- `configs/strategies/`：训练策略、优化器、调度器、预训练加载策略
- `configs/experiments/`：最终实验入口，使用 `_base_` 组合多个配置片段

## 下一步建议

- 增加 `builders/` 与更细的抽象接口，例如 `BaseDetector`、`BaseTracker`
- 引入真实的训练 `Runner`、Hook 机制与事件总线
- 为不同雷达任务补充标准数据适配器与评测协议
- 将单模型实验逐步扩展为多阶段流水线实验
