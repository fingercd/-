"""Evaluation helpers for fixed-window and streaming VAD predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..metrics import (
    UCFFrameMetrics,
    compute_ucf_frame_metrics,
    project_intervals_to_frames,
    resample_scores_to_frames,
)

try:  # Model-free artifact evaluation works without PyTorch.
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal environments.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class TemporalPrediction:
    """One video's temporal prediction with an optional explicit timeline."""

    video_id: str
    scores: Any
    intervals: Any | None = None
    fps: float | None = None
    frame_times: Any | None = None
    valid_mask: Any | None = None


@dataclass(frozen=True)
class UCFEvaluationResult:
    """Global UCF-Crime metrics plus the exact projected frame sequences."""

    metrics: UCFFrameMetrics
    frame_scores: dict[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return self.metrics.to_dict()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_numpy(value: Any) -> np.ndarray:
    if TORCH_AVAILABLE and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_parts(
    value: Any,
) -> tuple[np.ndarray, Any | None, float | None, Any | None, Any | None]:
    scores = _field(value, "scores")
    if scores is None:
        scores = _field(value, "frame_scores")
    if scores is None:
        scores = _field(value, "snippet_scores")
    if scores is None:
        predictions = _field(value, "predictions")
        if predictions is not None and predictions is not value:
            scores = _field(predictions, "snippet_scores", predictions)
    if scores is None:
        scores = value
    if TORCH_AVAILABLE and torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    score_array = np.asarray(scores)
    if score_array.ndim == 2 and score_array.shape[0] == 1:
        score_array = score_array[0]
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError(
            f"each prediction must contain a non-empty 1D score sequence, got {score_array.shape}"
        )
    if not np.all(np.isfinite(score_array)):
        raise ValueError("prediction contains NaN or infinite scores")
    auxiliary = _field(value, "auxiliary")
    valid_mask = _field(value, "valid_mask")
    if valid_mask is None and auxiliary is not None:
        valid_mask = _field(auxiliary, "valid_mask")
    intervals = _field(value, "intervals")
    if intervals is None and auxiliary is not None:
        timeline = _field(auxiliary, "timeline")
        if timeline is not None:
            if valid_mask is None:
                valid_mask = _field(timeline, "valid_mask", _field(timeline, "valid"))
            frame_start = _field(timeline, "source_frame_start")
            frame_end = _field(timeline, "source_frame_end")
            if frame_start is not None and frame_end is not None:
                starts = _to_numpy(frame_start)
                ends = _to_numpy(frame_end)
            else:
                starts = _to_numpy(_field(timeline, "start_s", _field(timeline, "start_seconds")))
                ends = _to_numpy(_field(timeline, "end_s", _field(timeline, "end_seconds")))
            if starts.ndim == 2 and starts.shape[0] == 1:
                starts = starts[0]
                ends = ends[0]
            if starts.shape != score_array.shape or ends.shape != score_array.shape:
                raise ValueError("prediction timeline must match its score sequence")
            intervals = np.column_stack((starts, ends))
    return (
        score_array,
        intervals,
        _field(value, "fps"),
        _field(value, "frame_times"),
        valid_mask,
    )


def project_video_prediction(
    prediction: Any,
    num_frames: int,
    *,
    intervals: Any | None = None,
    fps: float | None = None,
    frame_times: Any | None = None,
    reduction: str = "max",
    fill_value: float = 0.0,
    allow_uniform_resample: bool = True,
) -> np.ndarray:
    """Convert one frame/snippet/interval prediction to exactly ``num_frames``."""

    scores, embedded_intervals, embedded_fps, embedded_times, valid_mask = _prediction_parts(
        prediction
    )
    intervals = embedded_intervals if intervals is None else intervals
    fps = embedded_fps if fps is None else fps
    frame_times = embedded_times if frame_times is None else frame_times
    if valid_mask is not None:
        if TORCH_AVAILABLE and torch.is_tensor(valid_mask):
            valid_mask = valid_mask.detach().cpu().numpy()
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim == 2 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.shape != scores.shape:
            raise ValueError(
                f"prediction valid_mask must have shape {scores.shape}, got {valid.shape}"
            )
        scores = scores[valid]
        if scores.size == 0:
            raise ValueError("prediction valid_mask selects no scores")
        if intervals is not None:
            interval_array = np.asarray(intervals)
            if interval_array.shape != (valid.size, 2):
                raise ValueError("prediction intervals must match the unmasked score sequence")
            intervals = interval_array[valid]
    if intervals is not None:
        return project_intervals_to_frames(
            intervals,
            scores,
            num_frames,
            fps=fps,
            frame_times=frame_times,
            reduction=reduction,  # type: ignore[arg-type]
            fill_value=fill_value,
        )
    if scores.shape[0] == num_frames:
        return scores.astype(np.float64, copy=False)
    if not allow_uniform_resample:
        raise ValueError(f"got {scores.shape[0]} scores for {num_frames} frames and no intervals")
    return resample_scores_to_frames(scores, num_frames).astype(np.float64, copy=False)


def evaluate_ucf_predictions(
    predictions: Mapping[str, Any],
    frame_labels: Mapping[str, Any],
    *,
    intervals: Mapping[str, Any] | None = None,
    fps: Mapping[str, float] | float | None = None,
    frame_times: Mapping[str, Any] | None = None,
    reduction: str = "max",
    fill_value: float = 0.0,
    allow_uniform_resample: bool = True,
    undefined: Literal["nan", "raise"] = "nan",
) -> UCFEvaluationResult:
    """Run the canonical global UCF-Crime frame ROC-AUC/AP evaluation.

    Predictions may already be frame aligned or may provide interval/snippet
    scores.  Explicit recorded intervals are preferred; uniform expansion is
    retained for legacy 32-segment UCF feature files.
    """

    if not predictions or not frame_labels:
        raise ValueError("predictions and frame_labels must not be empty")
    if set(predictions) != set(frame_labels):
        raise ValueError(
            "prediction/label video ids differ: "
            f"prediction_only={sorted(set(predictions) - set(frame_labels))}, "
            f"label_only={sorted(set(frame_labels) - set(predictions))}"
        )

    projected: dict[str, np.ndarray] = {}
    normalized_labels: dict[str, np.ndarray] = {}
    for video_id, labels_value in frame_labels.items():
        labels = np.asarray(labels_value).reshape(-1)
        if labels.size == 0:
            raise ValueError(f"video {video_id!r} has no frame labels")
        normalized_labels[video_id] = labels
        video_intervals = None if intervals is None else intervals.get(video_id)
        video_times = None if frame_times is None else frame_times.get(video_id)
        video_fps = fps.get(video_id) if isinstance(fps, Mapping) else fps
        projected[video_id] = project_video_prediction(
            predictions[video_id],
            labels.size,
            intervals=video_intervals,
            fps=video_fps,
            frame_times=video_times,
            reduction=reduction,
            fill_value=fill_value,
            allow_uniform_resample=allow_uniform_resample,
        )

    metrics = compute_ucf_frame_metrics(normalized_labels, projected, undefined=undefined)
    return UCFEvaluationResult(metrics=metrics, frame_scores=projected)


def prediction_records_to_temporal(
    records: Iterable[Any],
) -> tuple[dict[str, TemporalPrediction], str]:
    """Group artifact ``PredictionRecord`` objects into temporal sequences.

    The function is structural to avoid coupling the evaluator to one storage
    backend.  It prefers recorded half-open frame ranges when every record has
    them, otherwise it uses ``start_s/end_s`` and reports ``"seconds"`` so the
    caller can provide FPS or exact frame timestamps.
    """

    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        video_id = _field(record, "video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("every prediction record must have a non-empty video_id")
        score = _field(record, "anomaly_score", _field(record, "score"))
        if score is None or not np.isfinite(float(score)):
            raise ValueError(f"prediction record for {video_id!r} has no finite score")
        grouped[video_id].append(record)
    if not grouped:
        raise ValueError("prediction records must not be empty")

    use_frames = all(
        _field(record, "frame_start") is not None and _field(record, "frame_end") is not None
        for video_records in grouped.values()
        for record in video_records
    )
    predictions: dict[str, TemporalPrediction] = {}
    for video_id, video_records in grouped.items():
        ordered = sorted(
            video_records,
            key=lambda record: (
                int(_field(record, "clip_index", 0)),
                float(_field(record, "start_s", 0.0)),
            ),
        )
        scores = np.asarray(
            [float(_field(record, "anomaly_score", _field(record, "score"))) for record in ordered],
            dtype=np.float64,
        )
        if use_frames:
            intervals = np.asarray(
                [
                    [int(_field(record, "frame_start")), int(_field(record, "frame_end"))]
                    for record in ordered
                ],
                dtype=np.int64,
            )
        else:
            intervals = np.asarray(
                [
                    [float(_field(record, "start_s")), float(_field(record, "end_s"))]
                    for record in ordered
                ],
                dtype=np.float64,
            )
        predictions[video_id] = TemporalPrediction(
            video_id=video_id, scores=scores, intervals=intervals
        )
    return predictions, "frames" if use_frames else "seconds"


def evaluate_ucf_prediction_records(
    records: Iterable[Any],
    frame_labels: Mapping[str, Any],
    *,
    fps: Mapping[str, float] | float | None = None,
    frame_times: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> UCFEvaluationResult:
    """Evaluate JSONL-style prediction records emitted by ``ArtifactStore``."""

    predictions, interval_unit = prediction_records_to_temporal(records)
    if interval_unit == "seconds" and fps is None and frame_times is None:
        raise ValueError(
            "second-based prediction records require fps or exact frame_times for projection"
        )
    return evaluate_ucf_predictions(
        predictions,
        frame_labels,
        fps=fps if interval_unit == "seconds" else None,
        frame_times=frame_times if interval_unit == "seconds" else None,
        **kwargs,
    )


def evaluate_ucf_frame_auc(
    predictions: Mapping[str, Any], frame_labels: Mapping[str, Any], **kwargs: Any
) -> float:
    """Return only the benchmark's primary frame ROC-AUC scalar."""

    return evaluate_ucf_predictions(predictions, frame_labels, **kwargs).metrics.frame_auc


def evaluate_batches(
    model: Any,
    batches: Iterable[Any],
    *,
    prediction_fn: Any | None = None,
    device: Any | None = None,
) -> list[Any]:
    """Run inference without gradients; projection/metrics remain separate.

    ``prediction_fn(output, batch)`` can turn task-specific outputs into
    :class:`TemporalPrediction` records. Tasks with ``prediction_step`` retain
    their timeline/mask automatically; otherwise raw outputs are returned.
    """

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to evaluate a model")
    if device is not None:
        model.to(device)
    model.eval()
    collected: list[Any] = []
    with torch.inference_mode():
        for batch in batches:
            if device is not None:
                from .train import move_to_device

                batch = move_to_device(batch, device)
            prediction_step = getattr(model, "prediction_step", None)
            output = prediction_step(batch) if callable(prediction_step) else model(batch)
            collected.append(prediction_fn(output, batch) if prediction_fn is not None else output)
    return collected


__all__ = [
    "TORCH_AVAILABLE",
    "TemporalPrediction",
    "UCFEvaluationResult",
    "evaluate_batches",
    "evaluate_ucf_frame_auc",
    "evaluate_ucf_prediction_records",
    "evaluate_ucf_predictions",
    "prediction_records_to_temporal",
    "project_video_prediction",
]
