from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


BASE_REPOS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
VOICE_DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
REQUIRED_FILES = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": [
        "config.json",
        "model.safetensors",
        "speech_tokenizer/model.safetensors",
    ],
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": [
        "config.json",
        "model.safetensors",
        "speech_tokenizer/model.safetensors",
    ],
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": [
        "config.json",
        "model.safetensors",
        "speech_tokenizer/model.safetensors",
    ],
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": [
        "config.json",
        "model.safetensors",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS model folders for ebook-tts.")
    parser.add_argument("--model-root", default="models/qwen-tts")
    parser.add_argument("--base-model", choices=sorted(BASE_REPOS), default="1.7B")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--disable-xet", action="store_true")
    args = parser.parse_args()

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    repos = [BASE_REPOS[args.base_model], VOICE_DESIGN_REPO, TOKENIZER_REPO]

    for repo_id in repos:
        download_repo(repo_id=repo_id, model_root=model_root, retries=args.retries)

    print(f"Qwen3-TTS models are ready under {model_root}")
    return 0


def download_repo(repo_id: str, model_root: Path, retries: int) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    local_dir = model_root / repo_id.split("/")[-1]
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {repo_id} -> {local_dir} (attempt {attempt}/{retries})")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            missing = missing_required_files(repo_id, local_dir)
            for filename in missing:
                print(f"Downloading missing required file {repo_id}/{filename}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
            missing = missing_required_files(repo_id, local_dir)
            if missing:
                raise RuntimeError(f"{repo_id} is missing required files: {missing}")
            return
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(5 * attempt)


def missing_required_files(repo_id: str, local_dir: Path) -> list[str]:
    missing = []
    for filename in REQUIRED_FILES.get(repo_id, []):
        path = local_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


if __name__ == "__main__":
    raise SystemExit(main())
