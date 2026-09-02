# Third-party provenance

| Component | Upstream | Revision | License retained at |
|---|---|---|---|
| SAM3 | `https://github.com/facebookresearch/sam3.git` | `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f` | `third_party/sam3/LICENSE` |
| Depth Anything 3 | `https://github.com/ByteDance-Seed/Depth-Anything-3.git` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `third_party/depth_anything_3/LICENSE` |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | model snapshot/model card |

The DA3 vendor tree has five documented global compatibility changes: explicit processing resolution, an upstream one-chunk artifact materialization path, a one-chunk full-frame save correction when streaming overlap is configured, a one-chunk camera-pose/intrinsics overlap correction, and numeric frame sorting. They apply uniformly to every input and are visible directly in `third_party/depth_anything_3/da3_streaming/da3_streaming.py`. SAM3 is unmodified upstream source.

Checkpoints are intentionally excluded from this source directory. Their expected SHA-256 hashes are recorded in `config.json` and checked during deployment.
