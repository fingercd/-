# VADBench

VADBench is a pluggable video-encoder, training, evaluation, and cache-compression research framework for UCF-Crime.

The first two reference paths are deliberately different:

- **VideoMAE V2 Base** is a stateless fixed-clip representation encoder. It has no reusable cross-clip `past_key_values` API.
- **HERMES + LLaVA-OneVision-Qwen2-0.5B** is a streaming VLM context method. It stores and compresses language-model decoder KV while exposing projected visual tokens for a classifier. It is not a visual encoder with native KV cache.

The framework provides versioned manifests, temporal provenance, fixed/streaming adapter contracts, content-addressed feature storage, weak MIL and explicit temporal supervision, frame-level ROC-AUC/AP, cache telemetry, pinned upstream revisions, and verified checkpoint downloads.

See the [full Chinese README](README-CN.md), [encoder survey](docs/research/video-encoder-survey-2026-08-31.md), [UCF-Crime protocol](docs/research/ucf-crime-protocol.md), and [implementation plan](docs/plans/2026-08-31-video-encoder-benchmark-framework.md).

## Quick start

```bash
uv sync --extra dev --extra train --extra video
uv run pytest
uv run vadbench doctor
```

Pin external source trees:

```bash
uv run python scripts/fetch_upstreams.py
uv run python scripts/fetch_upstreams.py --verify-only
```

Download checkpoints with explicit license acknowledgement:

```bash
uv run vadbench weights fetch videomaev2-base-hf weights/videomaev2-base-hf \
  --accept-license cc-by-nc-4.0
uv run vadbench weights fetch hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b \
  --accept-license apache-2.0
```

Import the official UCF-Crime split and temporal annotations:

```bash
uv run vadbench manifest import-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-split data/splits/Anomaly_Train.txt \
  --temporal-annotations data/splits/Temporal_Anomaly_Annotation.txt \
  --output-dir data/manifests/ucf_crime \
  --require-files --probe-video-info
```

The importer converts the official MATLAB 1-based inclusive coordinates into zero-based half-open spans: `165..240` becomes `[164,240)`, covering 76 frames.

Run a real-weight smoke test:

```bash
uv run vadbench smoke \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --video path/to/clip.mp4

uv run vadbench smoke \
  -c configs/experiments/ucf_hermes_stream.yaml \
  --video path/to/long-clip.mp4 --chunks 2
```

Large videos, weights, feature blobs, outputs, and external repositories are intentionally Git-ignored. The repository code is MIT-licensed; upstream code, datasets, and model weights retain their own licenses. In particular, the pinned VideoMAE V2 Base weights are CC-BY-NC-4.0.

