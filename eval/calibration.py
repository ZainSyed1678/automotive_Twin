"""
Probability calibration and uncertainty quantification for diagnostic models.
Computes Expected Calibration Error (ECE), Brier Score, and reliability diagram bins.
"""

from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

from casegen.schema import DifficultyLevel, GroundTruthCase
from .baselines import BaselinePrediction


class CalibrationBin(BaseModel):
    """Metrics for an individual probability confidence bin."""
    bin_lower: float
    bin_upper: float
    bin_label: str
    sample_count: int
    avg_confidence: float
    observed_accuracy: float
    calibration_gap: float


class CalibrationReport(BaseModel):
    """Comprehensive uncertainty calibration metrics."""
    mode: str
    total_eval_samples: int
    brier_score: float = Field(..., description="Mean squared difference between confidence and binary accuracy (lower is better, [0, 1])")
    expected_calibration_error: float = Field(..., description="Weighted average gap between confidence and observed accuracy (lower is better)")
    max_calibration_error: float = Field(..., description="Maximum absolute calibration gap across populated bins")
    reliability_bins: List[CalibrationBin] = Field(default_factory=list)


def compute_calibration_metrics(
    predictions: List[BaselinePrediction],
    ground_truth: List[GroundTruthCase],
    num_bins: int = 5,
) -> CalibrationReport:
    """
    Computes Brier Score and Expected Calibration Error (ECE) for diagnosable cases.
    Evaluates how reliably predicted posterior probabilities match true empirical diagnostic accuracy.
    """
    assert len(predictions) == len(ground_truth), "Mismatch between predictions and ground truth"

    mode = predictions[0].mode if predictions else "Unknown"

    # Filter strictly to diagnosable cases (exclude insufficient evidence where correct outcome is abstention)
    eval_pairs: List[Tuple[float, bool]] = []
    for pred, gt in zip(predictions, ground_truth):
        if gt.difficulty == DifficultyLevel.INSUFFICIENT_EVIDENCE:
            continue
        is_correct = (pred.top_fault_id == gt.true_fault_id)
        confidence = float(pred.confidence_score)
        eval_pairs.append((confidence, is_correct))

    n_samples = len(eval_pairs)
    if n_samples == 0:
        return CalibrationReport(
            mode=mode,
            total_eval_samples=0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
            reliability_bins=[],
        )

    # 1. Brier Score: (1/N) * sum((confidence - is_correct)^2)
    brier = sum((conf - (1.0 if correct else 0.0)) ** 2 for conf, correct in eval_pairs) / n_samples

    # 2. Equal-width binning for ECE
    bin_width = 1.0 / num_bins
    bins_data: List[CalibrationBin] = []
    weighted_ece = 0.0
    max_error = 0.0

    for b in range(num_bins):
        lower = b * bin_width
        upper = (b + 1) * bin_width
        # For the last bin, include the upper edge 1.0
        if b == num_bins - 1:
            bin_samples = [(c, y) for c, y in eval_pairs if lower <= c <= upper]
        else:
            bin_samples = [(c, y) for c, y in eval_pairs if lower <= c < upper]

        count = len(bin_samples)
        if count > 0:
            avg_conf = sum(c for c, _ in bin_samples) / count
            obs_acc = sum(1.0 for _, y in bin_samples if y) / count
            gap = abs(obs_acc - avg_conf)
            weighted_ece += (count / n_samples) * gap
            max_error = max(max_error, gap)
        else:
            avg_conf = 0.0
            obs_acc = 0.0
            gap = 0.0

        bins_data.append(
            CalibrationBin(
                bin_lower=round(lower, 2),
                bin_upper=round(upper, 2),
                bin_label=f"[{lower:.1f}, {upper:.1f}]",
                sample_count=count,
                avg_confidence=round(avg_conf, 4),
                observed_accuracy=round(obs_acc, 4),
                calibration_gap=round(gap, 4),
            )
        )

    return CalibrationReport(
        mode=mode,
        total_eval_samples=n_samples,
        brier_score=round(brier, 4),
        expected_calibration_error=round(weighted_ece, 4),
        max_calibration_error=round(max_error, 4),
        reliability_bins=bins_data,
    )
