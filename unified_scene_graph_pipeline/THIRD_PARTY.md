# Third-party provenance

| Component | Upstream | Revision | License retained at |
|---|---|---|---|
| SAM3 | `https://github.com/facebookresearch/sam3.git` | `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f` | `third_party/sam3/LICENSE` |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `third_party/sam2/LICENSE` |
| Depth Anything 3 | `https://github.com/ByteDance-Seed/Depth-Anything-3.git` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | `third_party/depth_anything_3/LICENSE` |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | model snapshot/model card |

The DA3 vendor tree has global compatibility changes for explicit processing
resolution, single-chunk artifact materialization, full-frame output with
streaming overlap, camera-pose/intrinsics handling, and numeric frame sorting.
The wrapper also regenerates a point cloud from valid native depth outputs when
DA3's confidence-filtered sample rounds down to zero points. The fallback keeps
all points passing DA3's native confidence rule and is recorded in the scene's
DA3 stage metadata. These rules apply uniformly to every input.

SAM3 and SAM2 are retained as unmodified upstream source. The DA3 changes are
visible in `third_party/depth_anything_3/da3_streaming/` and
`da3/container_scene.py`.

Checkpoints are intentionally excluded from this source directory. Their expected hashes or immutable revisions are recorded in the relevant pipeline configuration and checked at runtime.
