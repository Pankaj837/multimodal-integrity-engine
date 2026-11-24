"""Pre-download every model the pipeline uses (optional - they also download
automatically on first run). Useful before going offline or timing a run.

    python scripts/download_models.py

Models (all open weights, CPU-friendly):
  sentence-transformers/all-MiniLM-L6-v2   ~ 90 MB  embeddings
  Systran/faster-whisper-base              ~145 MB  speech recognition
  openai-community/gpt2                    ~550 MB  perplexity / burstiness
  Hello-SimpleAI/chatgpt-detector-roberta  ~500 MB  AI-text classifier
  Silero VAD                               bundled with faster-whisper
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integrity.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config(ROOT / "config.yaml").models
    from huggingface_hub import snapshot_download

    repos = [
        cfg.embedding,
        f"Systran/faster-whisper-{cfg.whisper}",
        cfg.ai_perplexity,
        cfg.ai_classifier,
    ]
    for repo in repos:
        print(f"Downloading {repo} ...", flush=True)
        # safetensors for PyTorch models; "model.bin" is the CTranslate2 Whisper
        # checkpoint (a generic *.bin would also pull redundant PyTorch pickles).
        path = snapshot_download(repo_id=repo, allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "model.bin", "vocabulary*", "tokenizer*", "merges.txt",
            "vocab*", "1_Pooling/*", "modules.json", "sentence_*",
        ])
        print(f"  -> {path}")
    print("All models are cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
