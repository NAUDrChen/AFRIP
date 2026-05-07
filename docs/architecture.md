# AFRIP 架构分层

## 1. 配置驱动

AFRIP 采用组合式配置：基础运行时、数据、检测器、跟踪器、训练策略彼此独立，在实验配置中通过 `_base_` 聚合。这种方式适合频繁切换模型、超参数与训练策略。

- `runtime` 仅承载设备、随机种子、日志和输出目录等运行环境参数
- `strategy.train` 统一承载训练轮数、混合精度、梯度裁剪、DataLoader worker 和训练恢复路径
- `strategy.eval` 统一承载评估 IoU、checkpoint 来源和权重输出目录
- `dataloader` 仅保留 `batch_size` 与 `shuffle`
- 后处理阈值和 NMS 策略统一放在 `detector.postprocessor_cfg`，检测模型主体不再声明这些后处理参数

## 2. 代码分层

- `core/`：只提供抽象基类、注册机制和无任务语义的 `build_from_config`
- `datasets/`：管理数据加载、增强、采样和批处理
- `models/`：作为模型域公共入口，统一提供模型侧注册表与 builder；`blocks/` 只承载纯神经网络原语
- `models/detection/`：承载检测域组件，包括 backbones、necks、heads、detectors、assigners、losses、preprocessors、postprocessors
- `models/tracking/`：承载跟踪域组件，包括 trackers 以及后续 motion、association、state estimator 等实现
- `engine/`：统一训练、评估和推理运行逻辑
- `strategies/`：组织优化器、调度器、损失权重策略、预训练加载策略
- `evaluation/`：沉淀任务指标、评测协议和分析工具
- `utils/`：放置配置、日志、随机数、路径等公共工具

## 3. 解耦原则

- 模型结构与训练策略分离
- 数据处理与模型主体分离
- 检测与跟踪既可独立运行，也可在实验层组合
- 基础模块通过注册器按需构建，避免硬编码依赖
- 检测模型装配优先通过 backbone / neck / head 三段注册配置完成，不再把特征图路由、组件引用和预测层拆成零碎图节点配置
- 脚本层只负责组装，不直接承载业务细节
- 跨模块公共接口优先使用普通 `dict[str, Tensor]`，避免为数据语义再叠加一层契约封装

## 4. 统一张量与坐标约定

### 4.1 图像张量

- 单样本图像统一为 `torch.float32`，形状 `[C, H, W]`
- 批量图像统一为 `torch.float32`，形状 `[B, C, H, W]`
- 数据集、增强、预处理、骨干网络、检测头和评估入口都使用这一表示

### 4.2 边界框与标签

- 公共边界框格式统一为绝对坐标 `xyxy`，即 `[x1, y1, x2, y2]`
- 边界框张量统一为 `torch.float32`，单样本形状 `[N, 4]`，模型输出形状 `[B, M, 4]` 或推理后 `[K, 4]`
- 标签统一为 `torch.int64`，单样本形状 `[N]`
- 分数统一为 `torch.float32`，推理后形状 `[K]`
- `xyxy` 坐标始终定义在“当前输入图像坐标系”下：训练窗口相对训练窗口，整帧验证相对整帧验证
- `x2`、`y2` 允许裁剪到图像边界，模块间不再传递归一化 `cxcywh`
- 若某个算法内部需要 `cxcywh`、中心点或网格坐标，应在模块内部局部转换，而不是把它作为跨模块公共接口

### 4.3 数据集与 batch 语义

- `Dataset.__getitem__` 返回以下字段：
	- `image`: `[C, H, W]`
	- `boxes`: `[N, 4]`，`xyxy`
	- `labels`: `[N]`
	- `meta`: 文件名、窗口原点、索引等非张量元信息
- `collate_fn` 返回以下字段：
	- `images`: `[B, C, H, W]`
	- `targets`: `list[dict]`，每项含 `boxes` 和 `labels`
	- `batch_meta`: 长度为 `B` 的元信息列表
- `batch` 不再额外暴露 `raw_boxes`、`batch_idx + box` 拼接张量或面向旧流程的框封装对象

### 4.4 模块边界张量约定

- `preprocessor` 输入统一为 `torch.Tensor[B, C, H, W]`，输出也必须是 `torch.Tensor[B, C, H, W]`
- `preprocessor` 只处理图像张量，不读取或改写 `boxes`、`labels`、`meta` 等监督信息
- `backbone` 输入统一为 `torch.Tensor[B, C, H, W]`
- `backbone` 输出允许为单个 `torch.Tensor`，也允许为按尺度排序的 `tuple[Tensor, ...]` / `list[Tensor]`；每个元素都必须保持批维在第 0 维、通道维在第 1 维
- `neck` 输入消费 `backbone` 的原始输出，输出统一为 `dict[str, torch.Tensor]`
- `neck` 输出字典的 key 表示语义特征层名，例如 `p2`、`p3`；value 统一为 `[B, C, H, W]`
- `head` 输入统一为 `dict[str, torch.Tensor]`，其中每个特征图都遵循 `[B, C, H, W]`
- `head` 训练态输出统一为普通 `dict`，当前检测主链至少包含：`pred_obj: [B, M, 1]`、`pred_box: [B, M, 4]`、`strides_all`、`fmp_sizes_all`
- `head` 输出的 `pred_box` 必须已经 decode 到当前输入图像坐标系下的绝对 `xyxy`，后续损失、匹配和评估不再重复 decode
- `postprocessor` 输入统一为单张图级别的 `boxes: [M, 4]` 和 `scores: [M]`，类型均为 `torch.Tensor`
- `postprocessor` 输出统一为普通 `dict`，包含 `boxes: [K, 4]`、`scores: [K]`、`labels: [K]`
- `postprocessor` 负责阈值过滤、框合法性过滤和 NMS；检测模型主体不再承载这些后处理细节
- `tracker` 当前仅保留弱接口约定：输入应基于单帧检测结果和必要的帧级上下文，输出应为带 track id 的时序结果
- 在 `tracker` 主链未稳定前，暂不把其内部状态张量形状、滤波状态维度或缓存结构写成强约束

### 4.5 训练态模型输出

- 训练态 `forward` 返回普通字典，不再返回额外包装对象
- 当前检测主链统一输出：
	- `pred_obj`: `[B, M, 1]`，objectness logits
	- `pred_box`: `[B, M, 4]`，已经 decode 到输入图像坐标系下的 `xyxy`
	- `strides_all`: 每个预测层的 stride 列表
	- `fmp_sizes_all`: 每个预测层的特征图尺寸列表
	- `stride`、`fmp_size`: 兼容单尺度损失读取的快捷字段

### 4.6 推理态输出

- 推理态 `forward` / `inference` 返回普通字典：
	- `boxes`: `[K, 4]`
	- `scores`: `[K]`
	- `labels`: `[K]`
- 后处理组件优先接收和返回 `torch.Tensor`
- `utils/nms.py` 内部仍兼容 `numpy` 输入，但这属于兼容实现细节，不是主流程接口标准

### 4.7 训练、分配、评估中的统一语义

- 标签分配器和损失函数直接消费 `targets[i]["boxes"]` 的 `xyxy` 表示
- 评估器直接消费 `batch["targets"]` 中的 `xyxy` GT 和模型推理返回的 `boxes`
- `obj_id`、`batch_idx`、原始文件坐标等信息不再混入公共 box tensor；需要时放入 `meta` 或可视化专用结构

## 5. 推荐扩展顺序

1. 定义 `BaseDataset`、`BaseDetector`、`BaseTracker` 抽象接口（已完成首版）
2. 增加 `builder` 与统一 `Runner`
3. 引入 Hook 机制，支持日志、评估、保存、可视化
4. 为典型雷达任务补齐数据协议与指标
5. 逐步支持联合训练、多阶段流水线与在线推理

## 6. 当前重构进展

- `core/base.py` 已提供 `BaseDataset`、`BaseDetector`、`BaseTracker`、`BaseModel`
- `models/registry.py` 已统一承载模型侧注册表；`models/blocks/` 已收敛为纯神经网络原语层
- `models/detection/` 已承载检测主链的 config-driven assembly、主干/颈部/检测头、assigner、loss、preprocessor 与 postprocessor
- `models/tracking/` 已预留为跟踪域入口，避免检测与跟踪继续混放在同一平面目录下
- `strategies/` 已提供正式 `build_optimizer`、`build_scheduler` 入口
- `strategies/` 已承接优化器与学习率调度器实现，优化器与调度器入口统一收口到 `afrip.strategies`
- `engine/Trainer` 已改为通过数据集实例解析 `collate_fn`，不再直接依赖 `RadarWindowDataset`
- 训练轮数仅从 `strategy.train.max_epoch` 读取，训练恢复与评估 checkpoint 已拆分为 `strategy.train.resume` 和 `strategy.eval.checkpoint`
- 检测后处理参数仅从 `detector.postprocessor_cfg` 读取，不再兼容 detector 顶层旧字段；检测器配置需显式声明 `preprocessor_cfg` 和 `postprocessor_cfg`
- 数据集、增强、训练、评估和后处理主链已统一为 `torch.Tensor + xyxy + plain dict` 接口
- `datasets` 层当前标准输出为 `image / boxes / labels / meta`，`collate_fn` 当前标准输出为 `images / targets / batch_meta`
- 检测器训练态输出和推理态输出都已切换为普通字典，不再在主链上传递额外契约对象
- 检测器训练/推理模式切换统一复用 PyTorch 原生 `train()/eval()`；`forward()` 基于 `nn.Module.training` 在训练原始输出与推理后处理结果之间切换
- `engine/runner.py` 已提供最小 `BaseRunner` / `DetectionRunner` 骨架，脚本入口已切换接入
- 已补充训练/验证 smoke test，覆盖两条主实验配置在统一接口上的一轮真实执行
