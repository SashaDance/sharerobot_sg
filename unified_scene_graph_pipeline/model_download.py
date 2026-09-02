"""Download the exact Qwen snapshot into a mounted model directory."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TARGET = Path("/models/qwen3.8-27b")


snapshot_download(
    repo_id=MODEL,
    revision=REVISION,
    local_dir=TARGET,
    token=True,
    max_workers=4,
)
(TARGET / "PINNED_REVISION").write_text(REVISION + "\n", encoding="utf-8")
print(f"Downloaded {MODEL} at {REVISION} to {TARGET}")
