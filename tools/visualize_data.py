from __future__ import annotations

from torch.utils.data import DataLoader

from afrip.datasets import build_transform_pipeline
from afrip.datasets.loaders.radar_window_dataset import RadarWindowDataset
from afrip.evaluation.visualize import visualize_batch_with_full, compute_and_visualize_stats

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置

if __name__ == "__main__":
    
    from afrip.datasets import build_transform_pipeline

    augment = build_transform_pipeline([
        {"type": "SeaClutterInjection", "prob": 1, "snr_db_range": [20.0, 21.0], "background_only": False},
        {"type": "RandomVerticalFlip", "prob": 1},
        {"type": "AmplitudeNormalize", "mode": "standard"},
    ])
    
    train_dataset = RadarWindowDataset(
        mat_dir="G:\Data\data",
        csv_path="G:\Data\data\data_mat.csv",
        window_size=(640, 640),
        stride=(640, 640),
        complex_mode="abs",
        class_mapping={"mt": 0, "wm": 1, "uk":2},
        cache_mat_files=8,
        transforms=None,
        subset='train',
        azimuth_split_ratio=0.7,
        full_frame=False
    )  
    # from utils.visualize import analyze_and_visualize_wm_diff
    # stats_wm = analyze_and_visualize_wm_diff(
    #     train_dataset,
    #     tolerance_pos=5.0,
    #     tolerance_size_ratio=0.5,
    #     reference_group=1,
    #     topn_frames=10,
    #     show=True
    # )

    # 统计与可视化
    # stats = compute_and_visualize_stats(train_dataset)
    loader = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=RadarWindowDataset.collate_fn)
    
    # import itertools

    # n = 10  # 想要的批次序号（从0开始）
    # batch = next(itertools.islice(loader, n, None))
    # visualize_batch_with_full(train_dataset, batch, complex_mode="abs", max_per_batch=8)


    for batch in loader:
            visualize_batch_with_full(train_dataset, batch, complex_mode="abs", max_per_batch=8)

