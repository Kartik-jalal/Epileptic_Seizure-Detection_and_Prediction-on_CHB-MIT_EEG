"""Locate the repo root and load config.yaml with all paths pre-anchored."""

from pathlib import Path
import yaml

# /Epileptic_Seizure-Detection_and_Prediction-on_CHB-MIT_EEG/src/es_detection_prediction/config.py -> parents[2] = /Epileptic_Seizure-Detection_and_Prediction-on_CHB-MIT_EEG/
REPO_ROOT = Path(__file__).resolve().parents[2]

def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    # Anchor every path in the config here,  once - so no consumer can
    # forget and silently resolve "./datasets/..." against its own cwd.
    for key, rel_path in config["data"]["dir"].items():
        config["data"]["dir"][key] = REPO_ROOT / rel_path
    return config  





