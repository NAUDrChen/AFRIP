# AFRIP 注册机制详解

## 1. 核心原理

AFRIP 的注册机制解决的核心问题是：**如何让一段 YAML 配置驱动 Python 对象的实例化，而不产生硬编码依赖**。

整个机制由三个层次组成：

```
core/registry.py          ← 通用注册器与构建函数（与业务无关）
datasets/registry.py      ← 数据层专属注册器实例（DATASETS、TRANSFORMS）
datasets/loaders/*.py     ← 具体类，通过装饰器写入注册器
datasets/transforms/*.py  ← 具体类，通过装饰器写入注册器
```

---

## 2. `Registry` 类的内部结构

```python
# src/afrip/core/registry.py

@dataclass
class Registry:
    name: str
    _items: dict[str, Callable] = field(default_factory=dict)
```

`_items` 就是一张普通字典，键是字符串名称，值是类（或任意可调用对象）：

```python
# 注册后 _items 的实际状态
DATASETS._items == {
    "RadarWindowDataset": <class 'RadarWindowDataset'>
}

TRANSFORMS._items == {
    "AmplitudeNormalize": <class 'AmplitudeNormalize'>,
    "RandomVerticalFlip": <class 'RandomVerticalFlip'>,
    "SeaClutterInjection": <class 'SeaClutterInjection'>,
    "PadToStride":         <class 'PadToStride'>,
}
```

---

## 3. `register()` 装饰器的工作原理

```python
def register(self, name: str | None = None):
    def decorator(target):
        key = name or target.__name__   # 未指定 name 则用类名
        if key in self._items:
            raise KeyError(...)         # 防止重复注册
        self._items[key] = target       # 写入字典
        return target                   # 原样返回类，不修改类本身
    return decorator
```

使用方式：

```python
# 方式 A：指定注册名（推荐，名称与类名解耦）
@TRANSFORMS.register("AmplitudeNormalize")
class AmplitudeNormalize:
    ...

# 方式 B：不指定，自动用类名
@DATASETS.register()
class RadarWindowDataset:
    ...
```

装饰器**不修改类本身**，只是在模块加载时把类的引用存入字典。

---

## 4. `build_from_config()` 的工作原理

```python
def build_from_config(config: dict, registry: Registry, **extra_kwargs) -> Any:
    target_type = config["type"]                          # 读取 type 字段
    kwargs = {k: v for k, v in config.items() if k != "type"}  # 其余字段作为参数
    kwargs.update(extra_kwargs)                           # 合并运行时额外参数
    return registry.get(target_type)(**kwargs)            # 查表 + 实例化
```

等价于手写：

```python
# YAML 配置
# type: AmplitudeNormalize
# mode: db_percentile

# build_from_config 做的事等价于：
AmplitudeNormalize(mode="db_percentile")
```

`extra_kwargs` 用于传入无法写进 YAML 的运行时对象，例如：

```python
# transforms 是运行时构建好的流水线对象，不能写进 YAML
build_dataset(config["dataset"], transforms=pipeline)
```

---

## 5. 注册触发时机

注册发生在**模块被首次导入时**，由 `datasets/__init__.py` 中的 side-effect import 负责触发：

```python
# src/afrip/datasets/__init__.py

from .registry import DATASETS, TRANSFORMS, ...   # 先创建注册器实例

from .loaders import radar_window_dataset          # 导入模块 → 执行装饰器 → 写入 DATASETS
from .transforms import radar_transforms           # 导入模块 → 执行装饰器 → 写入 TRANSFORMS
```

调用链如下：

```
import afrip.datasets
  └── 执行 datasets/__init__.py
        ├── 创建 DATASETS = Registry("datasets")
        ├── 创建 TRANSFORMS = Registry("transforms")
        ├── import radar_window_dataset
        │     └── @DATASETS.register("RadarWindowDataset") 执行
        │           └── DATASETS._items["RadarWindowDataset"] = RadarWindowDataset
        └── import radar_transforms
              ├── @TRANSFORMS.register("AmplitudeNormalize") 执行
              ├── @TRANSFORMS.register("RandomVerticalFlip") 执行
              ├── @TRANSFORMS.register("SeaClutterInjection") 执行
              └── @TRANSFORMS.register("PadToStride") 执行
```

任何导入了 `afrip.datasets` 的代码，注册器就已填充完毕，可以直接使用。

---

## 6. 完整的从 YAML 到对象实例的流程

以 `visualize_data.py` 为例：

```python
# 1. 加载 YAML 配置（config.py 处理 _base_ 继承并合并）
config = load_config("configs/datasets/radar_detection_sample.yaml")

# config["train_transforms"] 的值是：
# [
#   {"type": "SeaClutterInjection", "prob": 0.7, "snr_db_range": [5.0, 20.0], ...},
#   {"type": "RandomVerticalFlip", "prob": 0.5},
#   {"type": "AmplitudeNormalize", "mode": "db_percentile"},
#   {"type": "PadToStride", "stride": 32},
# ]

# 2. 构建 Transform 流水线
pipeline = build_transform_pipeline(config["train_transforms"])
# 内部逐条调用 build_from_config(cfg, TRANSFORMS)，最终得到 Compose([...])

# 3. 构建 Dataset（transforms 作为 extra_kwargs 注入）
dataset = build_dataset(config["dataset"], transforms=pipeline)
# 等价于：
# RadarWindowDataset(
#     mat_dir="G:/Data/data",
#     window_size=[512, 512],
#     stride=[256, 256],
#     ...,
#     transforms=pipeline   # ← extra_kwargs 注入
# )
```

---

## 7. 扩展：如何注册新组件

### 7.1 新增一个 Transform

```python
# src/afrip/datasets/transforms/my_transforms.py

from afrip.datasets.registry import TRANSFORMS
import torch, numpy as np

@TRANSFORMS.register("GaussianNoise")
class GaussianNoise:
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, image: torch.Tensor, raw_boxes: np.ndarray):
        noise = torch.randn_like(image) * self.sigma
        return image + noise, raw_boxes
```

在 `datasets/__init__.py` 中触发导入：

```python
from .transforms import my_transforms  # noqa: F401
```

在 YAML 中使用：

```yaml
train_transforms:
  - type: GaussianNoise
    sigma: 0.05
```

### 7.2 新增一个 Dataset

```python
# src/afrip/datasets/loaders/my_dataset.py

from afrip.datasets.registry import DATASETS
from torch.utils.data import Dataset

@DATASETS.register("MyRadarDataset")
class MyRadarDataset(Dataset):
    def __init__(self, data_dir: str, **kwargs):
        ...
```

在 YAML 中使用：

```yaml
dataset:
  type: MyRadarDataset
  data_dir: /path/to/data
```

### 7.3 新增其他层级的注册器（如 Models）

在对应包的 `registry.py` 中定义新注册器：

```python
# src/afrip/models/registry.py

from afrip.core import Registry, build_from_config

BACKBONES = Registry("backbones")
NECKS     = Registry("necks")
HEADS     = Registry("heads")
DETECTORS = Registry("detectors")
ASSIGNERS = Registry("assigners")
LOSSES    = Registry("losses")
PREPROCESSORS = Registry("preprocessors")
POSTPROCESSORS = Registry("postprocessors")
TRACKERS  = Registry("trackers")

def build_detector(config, **extra_kwargs):
    return build_from_config(config, DETECTORS, **extra_kwargs)

def build_assigner(config, **extra_kwargs):
    return build_from_config(config, ASSIGNERS, **extra_kwargs)
```

当前 `models/registry.py` 只维护任务域组件的注册器。`models/blocks/` 中的 `Conv`、`BasicBlock`、`Bottleneck` 等纯神经网络原语不再走 registry，而是直接通过 Python import 使用。

检测域配置中，标签分配器统一从 `loss.assigner_cfg` 构建；注册名当前使用 `YoloAssigner`、`SimOTAAssigner`。

---

## 8. 常见问题

### Q: 注册名与类名不一致时会怎样？

注册名由 `@registry.register("Name")` 中的字符串决定，与 Python 类名无关。YAML 里的 `type` 字段必须与**注册名**匹配，而不是类名。

```python
@TRANSFORMS.register("NormalizeAmplitude")   # 注册名
class AmplitudeNormalize:                     # 类名（无关）
    ...
```

```yaml
type: NormalizeAmplitude   # ✅ 匹配注册名
type: AmplitudeNormalize   # ❌ KeyError，匹配的是类名而非注册名
```

### Q: 忘记在 `__init__.py` 里触发导入会怎样？

装饰器只在模块被导入时执行。如果新文件没有被任何地方 import，注册不会发生，`build_from_config` 会抛出 `KeyError: "xxx is not registered in yyy"`。

解决方法：在对应包的 `__init__.py` 里加一行 side-effect import：

```python
from .loaders import my_new_loader  # noqa: F401
```

### Q: 同名注册会怎样？

`Registry.register()` 在键已存在时会抛出 `KeyError`，防止静默覆盖：

```
KeyError: "RadarWindowDataset is already registered in datasets"
```

### Q: 如何在运行时查看已注册了哪些组件？

```python
from afrip.datasets import DATASETS, TRANSFORMS

print(DATASETS.list())    # ['RadarWindowDataset']
print(TRANSFORMS.list())  # ['AmplitudeNormalize', 'PadToStride', 'RandomVerticalFlip', 'SeaClutterInjection']
```

或通过 CLI：

```bash
python -m afrip.cli show-config --config configs/experiments/detection/radardet_rdcnn_sort.yaml
```
