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
│  ├─ models/               # 模型域入口、任务域注册表、detection/tracking 子域
│  ├─ nn/                   # ultralytics 风格 parse_model 与最小图节点模块
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
- `configs/detectors/`：检测模型结构与超参数；结构主体由 `detector.model_cfg` 的 ultralytics 风格图配置描述
- `configs/trackers/`：跟踪模型结构与超参数
- `configs/strategies/`：训练策略、优化器、调度器、预训练加载策略
- `configs/experiments/`：最终实验入口，使用 `_base_` 组合多个配置片段
- 检测后处理配置统一放在 `detector.postprocessor_cfg`，标签分配器配置统一放在 `loss.assigner_cfg`
- 检测 graph 内部节点由 `afrip.nn.parse_model()` 解析；当前主链使用通用 `Detect` 产生原始预测，由 `DetectDecode` 负责通用 box decode，再由 `DetectContract` 适配到 AFRIP 既有训练/推理输出契约

## 下一步建议

- 扩充 `models/tracking/` 的真实实现，将 tracking 域从占位入口推进到可运行主链
- 在现有 `DetectionRunner` 基础上继续引入 Hook 机制与事件总线
- 为不同雷达任务补充标准数据适配器与评测协议
- 将单模型实验逐步扩展为多阶段流水线实验
