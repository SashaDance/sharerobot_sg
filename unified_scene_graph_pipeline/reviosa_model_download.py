from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "wooj0216/ReVIOSa-4B"
REVISION = "c8c66b578376dd5e8c70e5ddd6d4990e18dc9f5f"
TARGET = Path("/models/reviosa-4b")


marker = TARGET / ".snapshot_revision"
if TARGET.is_dir() and marker.is_file() and marker.read_text().strip() == REVISION:
    print(f"ReVIOSa snapshot is already present at {TARGET}")
else:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=TARGET,
        local_dir_use_symlinks=False,
    )
    (TARGET / ".snapshot_revision").write_text(f"{REVISION}\n")
    print(f"Downloaded {REPOSITORY}@{REVISION} to {TARGET}")
