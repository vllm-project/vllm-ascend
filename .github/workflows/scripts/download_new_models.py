import json
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download


def load_models_from_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return set(data.get("models", []))


def download_models(models: set[str]):
    for model in models:
        print(f"[download] Downloading model: {model}")
        snapshot_download(model_id=model)


def main(origin_config: str, new_config: str):
    previous_models = load_models_from_json(origin_config)
    current_models = load_models_from_json(new_config)

    new_models = current_models - previous_models
    if new_models:
        print(f"[info] Detected {len(new_models)} new model(s):")
        for model in new_models:
            print(f" - {model}")
        download_models(new_models)
    else:
        print("[info] No new models detected.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download new models from models_config.json")
    parser.add_argument(
        "--base-config",
        type=str,
        help="Path to the models configuration file",
    )
    parser.add_argument(
        "--new-config",
        type=str,
        default="models_config_new.json",
        help="Path to the new models configuration file",
    )
    args = parser.parse_args()

    main(args.config_file, args.config_file_new)
