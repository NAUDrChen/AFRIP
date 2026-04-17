from pathlib import Path

from afrip.utils import load_config


def test_experiment_config_inheritance() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / "radardet_rdcnn_sort.yaml"

    config = load_config(config_path)

    assert config["experiment"]["name"] == "radardet_rdcnn_sort"
    assert config["runtime"]["max_epochs"] == 24
    assert config["runtime"]["work_dir"] == "outputs/radardet_rdcnn_sort"
    assert config["detector"]["type"] == "RangeDopplerCNN"
    assert config["strategy"]["optimizer"]["lr"] == 0.0001
