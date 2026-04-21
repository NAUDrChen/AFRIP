from __future__ import annotations
import argparse
from torch.utils.data import DataLoader
from afrip.utils import load_config
from afrip.datasets import build_dataset, build_transform_pipeline
from afrip.evaluation.visualize import visualize_batch_with_full
from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./configs/datasets/radar_detection_sample.yaml', help="数据配置文件路径")
    args = parser.parse_args()

    import matplotlib
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']

    config = load_config(args.config)
    pipeline = build_transform_pipeline(config.get("train_transforms"))
    dataset = build_dataset(config["dataset"], transforms=pipeline)
    loader = DataLoader(dataset, collate_fn=RadarWindowDataset.collate_fn,
                        **{k: config["dataloader"][k] for k in ["batch_size", "shuffle"]},
                        num_workers=0)
    for batch in loader:
        visualize_batch_with_full(dataset, batch, complex_mode="abs")

if __name__ == "__main__":
    main()