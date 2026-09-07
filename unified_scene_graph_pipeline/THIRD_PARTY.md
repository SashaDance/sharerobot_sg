# Third-party provenance

| Component | Upstream | Revision | License retained at |
|---|---|---|---|
| SAM3 | `https://github.com/facebookresearch/sam3.git` | `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f` | `third_party/sam3/LICENSE` |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `third_party/sam2/LICENSE` |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | model snapshot/model card |

SAM3 and SAM2 are vendored without source modifications. Their checkpoints and
the Qwen model are excluded from Git. Expected checkpoint hashes and immutable
model revisions are recorded in `config.json` and checked at runtime.

Review and comply with the upstream licenses before use or redistribution.
