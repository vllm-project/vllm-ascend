import json
import subprocess
import tempfile
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

CONFIG_FILE = "models_config.json"


def load_models_from_file(file_path: Path) -> set[str]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return set(data.get("models", []))


def load_models_from_git(file_path: str, revision: str = "HEAD^") -> set[str]:
    try:
        content = subprocess.check_output(
            ["git", "show", f"{revision}:{file_path}"], text=True
        )
        data = json.loads(content)
        return set(data.get("models", []))
    except subprocess.CalledProcessError:
        print(f"[warn] Cannot find {file_path} in {revision}, assuming empty model list.")
        return set()
    except json.JSONDecodeError as e:
        print(f"[error] Failed to parse JSON from {revision}: {e}")
        return set()


def download_models(models: set[str]):
    for model in models:
        print(f"[download] Downloading model: {model}")
        snapshot_download(model_id=model, cache_dir="./modelscope_models")


def main():
    config_path = Path(CONFIG_FILE)

    if not config_path.exists():
        print(f"[error] Cannot find {CONFIG_FILE}")
        return

    current_models = load_models_from_file(config_path)
    previous_models = load_models_from_git(CONFIG_FILE)

    new_models = current_models - previous_models

    if new_models:
        print(f"[info] Detected {len(new_models)} new model(s):")
        for model in new_models:
            print(f" - {model}")
        download_models(new_models)
    else:
        print("[info] No new models detected.")


if __name__ == "__main__":
    main()
