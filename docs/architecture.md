# AFRIP 架构分层

## 1. 配置驱动

AFRIP 采用组合式配置：基础运行时、数据、检测器、跟踪器、训练策略彼此独立，在实验配置中通过 `_base_` 聚合。这种方式适合频繁切换模型、超参数与训练策略。

## 2. 代码分层

- `core/`：提供注册机制、构建逻辑和统一抽象
- `datasets/`：管理数据加载、增强、采样和批处理
- `models/`：承载骨干、颈部、检测头、匹配器、损失、跟踪器及 common 装配/契约模块
- `modules/`：容纳预处理、后处理、关联、滤波、神经网络等可重用模块
- `engine/`：统一训练、评估和推理运行逻辑
- `strategies/`：组织优化器、调度器、损失权重策略、预训练加载策略
- `evaluation/`：沉淀任务指标、评测协议和分析工具
- `utils/`：放置配置、日志、随机数、路径等公共工具

## 3. 解耦原则

- 模型结构与训练策略分离
- 数据处理与模型主体分离
- 检测与跟踪既可独立运行，也可在实验层组合
- 基础模块通过注册器按需构建，避免硬编码依赖
- 检测模型装配通过配置图定义特征流、组件引用和预测层，不再为不同 detector 版本维护独立 Python 类
- 脚本层只负责组装，不直接承载业务细节

## 4. 推荐扩展顺序

1. 定义 `BaseDataset`、`BaseDetector`、`BaseTracker` 抽象接口（已完成首版）
2. 增加 `builder` 与统一 `Runner`
3. 引入 Hook 机制，支持日志、评估、保存、可视化
4. 为典型雷达任务补齐数据协议与指标
5. 逐步支持联合训练、多阶段流水线与在线推理

## 5. 当前重构进展

- `core/base.py` 已提供 `BaseDataset`、`BaseDetector`、`BaseTracker`、`BaseModel`
- `models/common/` 已集中承载 blocks、registry、contracts 与 config-driven detection assembly
- `strategies/` 已提供正式 `build_optimizer`、`build_scheduler` 入口
- `strategies/` 已承接优化器与学习率调度器实现，`utils/solver` 仅保留兼容包装
- `engine/Trainer` 已改为通过数据集实例解析 `collate_fn`，不再直接依赖 `RadarWindowDataset`
- 检测器训练/推理模式切换已从隐式 `trainable` 属性迁移到显式接口
- `engine/runner.py` 已提供最小 `BaseRunner` / `DetectionRunner` 骨架，脚本入口已切换接入
