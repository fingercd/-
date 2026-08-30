from __future__ import annotations

import math
import unittest

import numpy as np

from vadbench.engine.evaluate import (
    evaluate_ucf_prediction_records,
    evaluate_ucf_predictions,
    prediction_records_to_temporal,
    project_video_prediction,
)
from vadbench.metrics import (
    average_precision_score,
    compute_ucf_frame_metrics,
    project_intervals_to_frames,
    project_intervals_to_grid,
    resample_scores_to_frames,
    roc_auc_score,
)


class BinaryMetricTests(unittest.TestCase):
    def test_roc_auc_and_average_precision_known_example(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc_score(labels, scores), 0.75)
        self.assertAlmostEqual(average_precision_score(labels, scores), 5.0 / 6.0)

    def test_metrics_are_tie_aware_and_order_invariant(self) -> None:
        labels = np.array([0, 1, 1, 0])
        tied = np.ones(4)
        self.assertAlmostEqual(roc_auc_score(labels, tied), 0.5)
        self.assertAlmostEqual(average_precision_score(labels, tied), 0.5)
        permutation = np.array([2, 0, 3, 1])
        self.assertEqual(
            average_precision_score(labels, tied),
            average_precision_score(labels[permutation], tied[permutation]),
        )

    def test_undefined_single_class_behavior(self) -> None:
        self.assertTrue(math.isnan(roc_auc_score([1, 1], [0.1, 0.2])))
        self.assertTrue(math.isnan(average_precision_score([0, 0], [0.1, 0.2])))
        with self.assertRaises(ValueError):
            roc_auc_score([1, 1], [0.1, 0.2], undefined="raise")

    def test_rejects_non_binary_or_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc_score([0, 2], [0.0, 1.0])
        with self.assertRaises(ValueError):
            average_precision_score([0, 1], [0.0, np.nan])


class ProjectionTests(unittest.TestCase):
    def test_grid_projection_overlap_reductions_and_gap(self) -> None:
        intervals = [[0.0, 2.0], [1.0, 3.0]]
        scores = [0.2, 0.8]
        grid = [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="max"),
            [0.2, 0.8, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="mean"),
            [0.2, 0.5, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="first"),
            [0.2, 0.2, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="last"),
            [0.2, 0.8, 0.8, 0.0],
        )

    def test_multichannel_mean_projection(self) -> None:
        result = project_intervals_to_grid(
            [[0, 2], [1, 3]],
            [[1.0, 3.0], [3.0, 5.0]],
            [0, 1, 2],
            reduction="mean",
        )
        np.testing.assert_allclose(result, [[1, 3], [2, 4], [3, 5]])

    def test_seconds_to_frame_projection_uses_half_open_intervals(self) -> None:
        result = project_intervals_to_frames(
            [[0.0, 1.0], [1.0, 2.0]], [0.1, 0.9], num_frames=4, fps=2.0
        )
        np.testing.assert_allclose(result, [0.1, 0.1, 0.9, 0.9])

    def test_uniform_resampling_is_piecewise_constant(self) -> None:
        np.testing.assert_array_equal(
            resample_scores_to_frames([1.0, 2.0], 5), [1.0, 1.0, 1.0, 2.0, 2.0]
        )
        np.testing.assert_array_equal(
            project_video_prediction([0.2, 0.8], 4), [0.2] * 2 + [0.8] * 2
        )

    def test_prediction_valid_mask_filters_padded_scores_and_intervals(self) -> None:
        result = project_video_prediction(
            {
                "scores": [0.1, 0.9, 100.0],
                "intervals": [[0, 2], [2, 4], [4, 6]],
                "valid_mask": [True, True, False],
            },
            4,
        )
        np.testing.assert_allclose(result, [0.1, 0.1, 0.9, 0.9])

    def test_task_prediction_timeline_supplies_frame_intervals(self) -> None:
        from types import SimpleNamespace

        prediction = SimpleNamespace(
            predictions=np.array([[0.1, 0.9, 100.0]]),
            auxiliary={
                "timeline": SimpleNamespace(
                    source_frame_start=np.array([[0, 2, 4]]),
                    source_frame_end=np.array([[2, 4, 6]]),
                    valid_mask=np.array([[True, True, False]]),
                )
            },
        )
        np.testing.assert_allclose(project_video_prediction(prediction, 4), [0.1, 0.1, 0.9, 0.9])

    def test_invalid_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_intervals_to_grid([[1, 1]], [0.5], [1])


class UCFProtocolTests(unittest.TestCase):
    def test_global_frame_metrics_concatenate_videos(self) -> None:
        labels = {"normal": np.array([0, 0]), "abnormal": np.array([1, 1])}
        scores = {"normal": np.array([0.1, 0.2]), "abnormal": np.array([0.8, 0.9])}
        result = compute_ucf_frame_metrics(labels, scores)
        self.assertEqual(result.num_videos, 2)
        self.assertEqual(result.num_frames, 4)
        self.assertEqual(result.num_positive_frames, 2)
        self.assertAlmostEqual(result.frame_auc, 1.0)
        self.assertAlmostEqual(result.frame_ap, 1.0)

    def test_variable_length_video_sequences_are_supported(self) -> None:
        labels = [np.array([0]), np.array([0, 1, 1])]
        scores = [np.array([0.1]), np.array([0.2, 0.8, 0.9])]
        result = compute_ucf_frame_metrics(labels, scores)
        self.assertEqual(result.num_videos, 2)
        self.assertEqual(result.num_frames, 4)
        self.assertAlmostEqual(result.frame_auc, 1.0)

    def test_ucf_evaluation_projects_recorded_intervals(self) -> None:
        labels = {"v": np.array([0, 0, 1, 1])}
        predictions = {
            "v": {
                "scores": np.array([0.1, 0.9]),
                "intervals": np.array([[0, 2], [2, 4]]),
            }
        }
        result = evaluate_ucf_predictions(predictions, labels)
        np.testing.assert_allclose(result.frame_scores["v"], [0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(result.metrics.frame_auc, 1.0)
        self.assertAlmostEqual(result.metrics.frame_ap, 1.0)

    def test_artifact_prediction_records_use_frame_ranges(self) -> None:
        from types import SimpleNamespace

        records = [
            SimpleNamespace(
                video_id="v",
                clip_index=1,
                frame_start=2,
                frame_end=4,
                start_s=1.0,
                end_s=2.0,
                anomaly_score=0.9,
            ),
            SimpleNamespace(
                video_id="v",
                clip_index=0,
                frame_start=0,
                frame_end=2,
                start_s=0.0,
                end_s=1.0,
                anomaly_score=0.1,
            ),
        ]
        grouped, interval_unit = prediction_records_to_temporal(records)
        self.assertEqual(interval_unit, "frames")
        np.testing.assert_allclose(grouped["v"].scores, [0.1, 0.9])
        result = evaluate_ucf_prediction_records(records, {"v": [0, 0, 1, 1]})
        self.assertAlmostEqual(result.metrics.frame_auc, 1.0)

    def test_second_records_require_timing_information(self) -> None:
        from types import SimpleNamespace

        records = [
            SimpleNamespace(
                video_id="v",
                clip_index=0,
                frame_start=None,
                frame_end=None,
                start_s=0.0,
                end_s=1.0,
                anomaly_score=0.2,
            )
        ]
        with self.assertRaises(ValueError):
            evaluate_ucf_prediction_records(records, {"v": [0, 0]})

    def test_mapping_key_and_length_mismatch_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_ucf_frame_metrics({"a": [0]}, {"b": [0.1]})
        with self.assertRaises(ValueError):
            compute_ucf_frame_metrics({"a": [0, 1]}, {"a": [0.1]})


if __name__ == "__main__":
    unittest.main()
