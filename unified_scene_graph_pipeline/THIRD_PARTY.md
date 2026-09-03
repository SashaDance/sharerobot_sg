# Third-party provenance

| Component | Upstream | Revision | License retained at |
|---|---|---|---|
| SAM3 | `https://github.com/facebookresearch/sam3.git` | `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f` | `third_party/sam3/LICENSE` |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `third_party/sam2/LICENSE` |
| RobotSeg | `https://github.com/showlab/RobotSeg.git` | `dafb8c0d507276e2f96d2b07ac3661a7b3a41a5f` | `third_party/robotseg/LICENSE` |
| InterRVOS/ReVIOSa | `https://github.com/cvlab-kaist/InterRVOS.git` | `25fb94bcf1159d4839607c8f829af7a55dfc335e` | `third_party/interrvos/LICENSE` |
| Depth Anything 3 | `https://github.com/ByteDance-Seed/Depth-Anything-3.git` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `third_party/depth_anything_3/LICENSE` |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | model snapshot/model card |

The DA3 vendor tree has five documented global compatibility changes: explicit processing resolution, an upstream one-chunk artifact materialization path, a one-chunk full-frame save correction when streaming overlap is configured, a one-chunk camera-pose/intrinsics overlap correction, and numeric frame sorting. They apply uniformly to every input and are visible directly in `third_party/depth_anything_3/da3_streaming/da3_streaming.py`. SAM3, SAM2, and RobotSeg are unmodified upstream source. RobotSeg's large README media assets are omitted from the vendor tree; its source, tests, metadata, and Apache-2.0 license are retained.

The experimental ReVIOSa checkpoint is pinned separately to Hugging Face revision `c8c66b578376dd5e8c70e5ddd6d4990e18dc9f5f`; its author-provided Docker image is pinned to manifest digest `sha256:6405558c773b1bdb3251dd6db65bbc975147e6533e9ab447782caac28c9eb8ac`.

Checkpoints are intentionally excluded from this source directory. Their expected hashes or immutable revisions are recorded in the relevant pipeline configuration and checked at runtime.
