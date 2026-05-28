from pathlib import Path

from afrip.utils import load_config


def test_experiment_config_inheritance() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / "radardet_rdcnn_sort.yaml"

    config = load_config(config_path)

    assert config["experiment"]["name"] == "radardet_rdcnn_sort"
    assert config["runtime"]["work_dir"] == "outputs/radardet_rdcnn_sort"
    assert "max_epochs" not in config["runtime"]
    assert config["detector"]["type"] == "DetectionModel"
    assert "conf_thresh" not in config["detector"]
    assert "nms_thresh" not in config["detector"]
    assert "trainable" not in config["detector"]
    assert "deploy" not in config["detector"]
    assert "model_cfg" in config["detector"]
    assert "backbone_cfg" not in config["detector"]
    assert "neck_cfg" not in config["detector"]
    assert "head_cfg" not in config["detector"]
    assert "component_cfgs" not in config["detector"]
    assert config["detector"]["model_cfg"]["head"][-3][2] == "Detect"
    assert config["detector"]["model_cfg"]["head"][-2][2] == "DetectDecode"
    assert config["detector"]["model_cfg"]["head"][-1][2] == "DetectContract"
    assert config["detector"]["postprocessor_cfg"]["conf_thresh"] == 0.01
    assert config["detector"]["postprocessor_cfg"]["nms_thresh"] == 0.5
    assert config["strategy"]["optimizer"]["lr0"] == 0.0001
    assert config["strategy"]["train"]["max_epoch"] == 50
    assert config["strategy"]["train"]["resume"] is None
    assert config["strategy"]["eval"]["checkpoint"] is None


def test_multiscale_detector_uses_assigner_cfg() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / "radardet_yolortv2_sort.yaml"

    config = load_config(config_path)

    assert "matcher_cfg" not in config["loss"]
    assert "assigner_cfg" in config["loss"]
    assert config["loss"]["assigner_cfg"]["type"] == "SimOTAAssigner"
    assert config["detector"]["model_cfg"]["head"][-3][2] == "Detect"
    assert config["detector"]["model_cfg"]["head"][-2][2] == "DetectDecode"
    assert config["detector"]["model_cfg"]["head"][-1][2] == "DetectContract"


def test_yolo26_config_uses_task_aligned_loss() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "experiments" / "detection" / "radardet_yolo26_sort.yaml"

    config = load_config(config_path)

    assert config["experiment"]["name"] == "radardet_yolo26_sort"
    assert config["detector"]["model_cfg"]["head"][-3][2] == "Detect"
    assert config["detector"]["model_cfg"]["head"][-2][2] == "DetectDecode"
    assert config["detector"]["model_cfg"]["head"][-1][2] == "DetectContract"
    assert config["detector"]["model_cfg"]["head"][-3][3][-1] is True
    assert config["loss"]["type"] == "Yolo26Criterion"
    assert config["loss"]["assigner_cfg"]["type"] == "TaskAlignedAssigner"
    assert config["loss"]["one2one_assigner_cfg"]["type"] == "TaskAlignedAssigner"
    assert config["val_transforms"][-1]["type"] == "PadToStride"
    assert config["val_transforms"][-1]["stride"] == 32
