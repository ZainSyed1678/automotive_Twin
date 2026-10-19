"""
Tests for Milestone 7: Probability calibration, Brier score, ECE, and held-out evaluation.
"""

import pytest
from casegen.schema import DifficultyLevel, GroundTruthCase, Observations
from eval.baselines import BaselinePrediction
from eval.calibration import compute_calibration_metrics, CalibrationReport
from eval.evaluate_heldout import run_heldout_evaluation


def test_brier_score_perfect_prediction():
    """Verify Brier score is 0.0 for perfectly calibrated, 100% accurate predictions."""
    gt_cases = [
        GroundTruthCase(
            case_id=f"CASE-{i}",
            true_fault_id="fuel_pump_weak",
            difficulty=DifficultyLevel.STRAIGHTFORWARD,
            observations=Observations(),
        )
        for i in range(5)
    ]
    predictions = [
        BaselinePrediction(
            case_id=f"CASE-{i}",
            top_fault_id="fuel_pump_weak",
            confidence_score=1.0,
            ranked_candidates=["fuel_pump_weak"],
            mode="Full-System",
        )
        for i in range(5)
    ]

    report: CalibrationReport = compute_calibration_metrics(predictions, gt_cases)
    assert report.brier_score == 0.0
    assert report.expected_calibration_error == 0.0


def test_brier_score_worst_prediction():
    """Verify Brier score is 1.0 when confident predictions are completely wrong."""
    gt_cases = [
        GroundTruthCase(
            case_id="CASE-1",
            true_fault_id="intake_vacuum_leak",
            difficulty=DifficultyLevel.STRAIGHTFORWARD,
            observations=Observations(),
        )
    ]
    predictions = [
        BaselinePrediction(
            case_id="CASE-1",
            top_fault_id="fuel_pump_weak",  # Wrong!
            confidence_score=1.0,            # But 100% confident!
            ranked_candidates=["fuel_pump_weak"],
            mode="Full-System",
        )
    ]

    report: CalibrationReport = compute_calibration_metrics(predictions, gt_cases)
    assert report.brier_score == 1.0


def test_ece_bins_calculation():
    """Verify Expected Calibration Error binning and weighted gap logic."""
    gt_cases = [
        GroundTruthCase(case_id="C1", true_fault_id="F1", difficulty=DifficultyLevel.STRAIGHTFORWARD, observations=Observations()),
        GroundTruthCase(case_id="C2", true_fault_id="F1", difficulty=DifficultyLevel.STRAIGHTFORWARD, observations=Observations()),
        GroundTruthCase(case_id="C3", true_fault_id="F2", difficulty=DifficultyLevel.STRAIGHTFORWARD, observations=Observations()),
        GroundTruthCase(case_id="C4", true_fault_id="F2", difficulty=DifficultyLevel.STRAIGHTFORWARD, observations=Observations()),
    ]
    # Two correct with 0.8 conf, two wrong with 0.8 conf -> Observed accuracy in [0.8, 1.0] bin is 50%, conf is 80% -> gap = 0.30
    predictions = [
        BaselinePrediction(case_id="C1", top_fault_id="F1", confidence_score=0.8, mode="Test"),
        BaselinePrediction(case_id="C2", top_fault_id="F1", confidence_score=0.8, mode="Test"),
        BaselinePrediction(case_id="C3", top_fault_id="F1", confidence_score=0.8, mode="Test"),  # Wrong
        BaselinePrediction(case_id="C4", top_fault_id="F1", confidence_score=0.8, mode="Test"),  # Wrong
    ]

    report = compute_calibration_metrics(predictions, gt_cases, num_bins=5)
    assert pytest.approx(report.expected_calibration_error, 1e-3) == 0.30
    assert len(report.reliability_bins) == 5


def test_heldout_evaluation_runner_execution():
    """Verify that run_heldout_evaluation executes cleanly and produces structured results."""
    results = run_heldout_evaluation(
        eval_cases_path="data/cases_eval.json",
        output_path="data/final_evaluation_m7.json",
    )

    assert "comparative_metrics" in results
    assert "calibration_reports" in results
    assert "adaptive_questioning_gains" in results

    metrics = results["comparative_metrics"]
    assert len(metrics) == 3

    full_system = next(m for m in metrics if m["mode"] == "Full-System")
    assert full_system["top1_accuracy"] > 0.70
    assert full_system["insufficient_evidence_accuracy"] == 1.0
    assert full_system["unsupported_claim_rate"] == 0.0
    assert full_system["brier_score"] < 0.10
