from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
    path.chmod(0o644)


def numeric_key(path: Path) -> tuple[int, str]:
    match = FRAME_RE.search(path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        [*directory.glob("*.png"), *directory.glob("*.jpg"), *directory.glob("*.jpeg")],
        key=numeric_key,
    )
    if not paths:
        raise ValueError(f"No images under {directory}")
    return paths


def normalized_frame_name(index: int) -> str:
    return f"frame_{index:06d}.png"


def sha256_path(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def chunks(values: list[int], size: int) -> Iterable[tuple[list[int], int | None]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, len(values), size):
        current = values[start : start + size]
        yield current, (start - 1 if start else None)


def mask_geometry(mask_path: Path) -> dict[str, Any]:
    import numpy as np

    with Image.open(mask_path) as image:
        pixels = np.asarray(image.convert("L")) > 0
    ys, xs = np.nonzero(pixels)
    if not len(xs):
        return {"status": "not_visible", "area": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "status": "visible",
        "area": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_xy": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
    }


def mask_label_placement(mask: Any) -> tuple[int, int, float]:
    """Place a marker deep inside a mask and return its local clearance radius."""
    import cv2
    import numpy as np

    pixels = np.asarray(mask, dtype=np.uint8)
    if pixels.ndim != 2 or not pixels.any():
        raise ValueError("Marker placement requires a non-empty 2D mask")
    distance = cv2.distanceTransform(pixels, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return int(x), int(y), float(distance[y, x])


def ensure_no_confidence(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "confidence" in key.lower():
                raise ValueError(f"Forbidden confidence field at {location}.{key}")
            ensure_no_confidence(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            ensure_no_confidence(item, f"{location}[{index}]")


def require_api_key() -> str:
    key = os.environ.get("INFERENCE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("INFERENCE_API_KEY is not set")
    return key


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stage_fingerprint(config: dict[str, Any], inputs: Iterable[Path], stage: str) -> str:
    digest = hashlib.sha256()
    digest.update(stage.encode())
    digest.update(config_hash(config).encode())
    for path in sorted(inputs, key=lambda item: str(item)):
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(sha256_path(path).encode())
        elif path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                digest.update(str(child.relative_to(path)).encode())
                digest.update(sha256_path(child).encode())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def stage_current(output: Path, stage: str, fingerprint: str, artifacts: Iterable[Path]) -> bool:
    marker = output / ".stages" / f"{stage}.json"
    if not marker.is_file() or any(not path.exists() for path in artifacts):
        return False
    try:
        return read_json(marker).get("fingerprint") == fingerprint
    except Exception:
        return False


def complete_stage(output: Path, stage: str, fingerprint: str, details: dict[str, Any] | None = None) -> None:
    write_json_atomic(
        output / ".stages" / f"{stage}.json",
        {"stage": stage, "fingerprint": fingerprint, "completed_at": utc_now(), **(details or {})},
    )


def update_run_report(output: Path, stage: str, status: str, details: dict[str, Any] | None = None) -> None:
    path = output / "run_report.json"
    report = read_json(path) if path.is_file() else {
        "schema_version": "unified_sgg_run_report_v1",
        "created_at": utc_now(),
        "stages": {},
    }
    report["updated_at"] = utc_now()
    report["stages"][stage] = {"status": status, "updated_at": utc_now(), **(details or {})}
    write_json_atomic(path, report)
