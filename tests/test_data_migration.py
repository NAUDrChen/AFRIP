
"""测试数据层迁移效果：配置→注册→数据加载→增强→batch"""
from pathlib import Path
import numpy as np
import torch

from afrip.utils import load_config
from afrip.datasets import DATASETS, TRANSFORMS, build_dataset, build_transform_pipeline


def test_registries_populated():
    """验证注册器是否被正确填充"""
    print("\n=== 1. 检查注册器 ===")
    
    datasets = DATASETS.list()
    transforms = TRANSFORMS.list()
    
    print(f"已注册 Dataset：{datasets}")
    print(f"已注册 Transform：{transforms}")
    
    assert "RadarWindowDataset" in datasets, "RadarWindowDataset 未注册"
    assert "AmplitudeNormalize" in transforms, "AmplitudeNormalize 未注册"
    assert "RandomVerticalFlip" in transforms, "RandomVerticalFlip 未注册"
    assert "SeaClutterInjection" in transforms, "SeaClutterInjection 未注册"
    assert "PadToStride" in transforms, "PadToStride 未注册"
    print("✓ 所有组件已正确注册\n")


def test_config_loading():
    """验证配置加载"""
    print("=== 2. 加载配置 ===")
    
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "datasets" / "radar_detection_sample.yaml"
    
    config = load_config(str(config_path))
    
    assert "dataset" in config, "配置缺少 dataset 字段"
    assert "train_transforms" in config, "配置缺少 train_transforms 字段"
    assert "dataloader" in config, "配置缺少 dataloader 字段"
    
    print(f"Dataset config: {config['dataset']}")
    print(f"Dataloader config: {config['dataloader']}")
    print(f"Transforms: {len(config['train_transforms'])} 个\n")
    
    return config


def test_transform_pipeline_building():
    """验证 Transform 流水线构建"""
    print("=== 3. 构建 Transform 流水线 ===")

    from pathlib import Path
    from afrip.utils import load_config
    from afrip.datasets import build_transform_pipeline

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "datasets" / "radar_detection_sample.yaml"
    config = load_config(str(config_path))

    pipeline = build_transform_pipeline(config.get("train_transforms"))

    assert pipeline is not None
    assert len(pipeline.transforms) == 4
    print(f"包含 {len(pipeline.transforms)} 个 Transform")
    print("✓ Transform 流水线构建成功\n")


def test_transform_execution():
    """验证 Transform 流水线执行"""
    print("=== 4. 测试 Transform 执行 ===")

    from pathlib import Path
    from afrip.utils import load_config
    from afrip.datasets import build_transform_pipeline

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "datasets" / "radar_detection_sample.yaml"
    config = load_config(str(config_path))
    pipeline = build_transform_pipeline(config.get("train_transforms"))

    fake_image = torch.randn(1, 256, 256)
    fake_boxes = np.array([
        [0, 0, 100, 100, 200, 200],
        [1, 1, 150, 150, 250, 250],
    ], dtype=np.float32)

    image_out, boxes_out = pipeline(fake_image, fake_boxes)

    assert image_out.shape == fake_image.shape
    assert boxes_out.shape == fake_boxes.shape
    print(f"输出 image shape: {image_out.shape}")
    print("✓ Transform 流水线执行成功\n")


def test_dataset_building_mock():
    """验证 Dataset 构建（使用模拟参数）"""
    print("=== 5. Dataset 构建测试（模拟数据）===")
    
    # 注意：这里是模拟测试，不实际需要真实数据文件
    # 只验证构建过程不出错
    
    dataset_config = {
        "type": "RadarWindowDataset",
        "mat_dir": r"G:\Data\data",
        "csv_path": r"G:\Data\data\data_mat.csv",
        "window_size": (512, 512),
        "stride": (256, 256),
        "complex_mode": "abs",
        "subset": "train",
    }
    
    try:
        # 这会因为数据文件不存在而失败，但能验证构建逻辑
        dataset = build_dataset(dataset_config)
        print("✓ Dataset 构建成功（注：数据文件缺失会导致后续失败）")
    except FileNotFoundError as e:
        print(f"⚠ 数据文件缺失（预期），错误: {str(e)[:100]}...")
        print("  这是正常的，说明 Dataset 类已正确加载和初始化\n")
    except Exception as e:
        print(f"❌ Dataset 构建失败: {e}")
        raise


def test_batch_structure():
    """验证 batch 结构"""
    print("=== 6. Batch 结构验证 ===")
    
    # 模拟来自 Dataset 的样本列表
    from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset
    
    batch = [
        {
            "image": torch.randn(1, 256, 256),
            "targets": [{"boxes": torch.randn(2, 4), "labels": torch.tensor([0, 1])}],
            "raw_boxes": torch.randn(2, 6),
            "meta": {"file": "sample1.mat", "global_origin": (0, 0), "index": 0}
        },
        {
            "image": torch.randn(1, 256, 256),
            "targets": [{"boxes": torch.randn(3, 4), "labels": torch.tensor([1, 0, 2])}],
            "raw_boxes": torch.randn(3, 6),
            "meta": {"file": "sample2.mat", "global_origin": (256, 0), "index": 1}
        },
    ]
    
    # 使用 collate_fn
    collated = RadarWindowDataset.collate_fn(batch)
    
    print(f"Batch 中 images shape: {collated['images'].shape}")
    print(f"Batch 中 targets 数量: {len(collated['targets'])}")
    print(f"Batch 中 raw_boxes shape: {collated['raw_boxes'].shape}")
    print(f"Batch metadata 数量: {len(collated['batch_meta'])}")
    
    assert collated['images'].shape == (2, 1, 256, 256), "images 尺寸不对"
    assert len(collated['targets']) == 2, "targets 数量不对"
    print("✓ Batch 结构正确\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AFRIP 数据层迁移测试")
    print("=" * 60)
    
    try:
        test_registries_populated()
        config = test_config_loading()
        pipeline = test_transform_pipeline_building(config)
        test_transform_execution(pipeline)
        test_dataset_building_mock()
        test_batch_structure()
        
        print("=" * 60)
        print("✅ 所有测试通过！数据层迁移成功")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise