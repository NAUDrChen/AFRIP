from pathlib import Path

from afrip.utils import load_config


def test_experiment_config_inheritance() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / "radardet_rdcnn_sort.yaml"

    config = load_config(config_path)

    assert config["experiment"]["name"] == "radardet_rdcnn_sort"
    assert config["runtime"]["work_dir"] == "outputs/radardet_rdcnn_sort"
    assert "max_epochs" not in config["runtime"]
    assert config["detector"]["type"] == "ConfigurableDetectionModel"
    assert "conf_thresh" not in config["detector"]
    assert "nms_thresh" not in config["detector"]
    assert "neck_cfg" in config["detector"]
    assert "head_cfg" in config["detector"]
    assert "component_cfgs" not in config["detector"]
    assert config["detector"]["postprocessor_cfg"]["conf_thresh"] == 0.01
    assert config["detector"]["postprocessor_cfg"]["nms_thresh"] == 0.5
    assert config["strategy"]["optimizer"]["lr0"] == 0.0001
    assert config["strategy"]["train"]["max_epoch"] == 50
    assert config["strategy"]["train"]["resume"] is None
    assert config["strategy"]["eval"]["checkpoint"] is None
