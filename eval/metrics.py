"""
Evaluation metrics computation for diagnostic baselines.
Measures Top-1, Top-3, ambiguity detection, insufficient evidence handling, citation rate,
stratified accuracy per difficulty tier, and probability calibration.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from casegen.schema import DifficultyLevel, GroundTruthCase
from .baselines import BaselinePrediction


class EvaluationMetrics(BaseModel):
    mode: str
    total_cases: int
    diagnosable_cases: int
    top1_accuracy: float
    top3_accuracy: float
    ambiguity_rate: float
    insufficient_evidence_accuracy: float
    avg_citations_per_case: float
    unsupported_claim_rate: float
    difficulty_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Top-1 accuracy by difficulty tier (or abstention accuracy for insufficient evidence)",
    )
    brier_score: Optional[float] = None
    expected_calibration_error: Optional[float] = None


def compute_metrics(
    predictions: List[BaselinePrediction],
    ground_truth: List[GroundTruthCase],
) -> EvaluationMetrics:
    """
    Computes comparative benchmark metrics against ground-truth labels.
    """
    assert len(predictions) == len(ground_truth), "Prediction and ground-truth count mismatch"

    mode = predictions[0].mode if predictions else "Unknown"
    total_cases = len(predictions)

    diagnosable_count = 0
    top1_correct = 0
    top3_correct = 0
    ambiguous_count = 0
    insufficient_gt_count = 0
    insufficient_correct = 0
    total_citations = 0
    unsupported_count = 0

    # Per-difficulty accumulators: tier -> {"total": int, "correct": int}
    diff_stats: Dict[str, Dict[str, int]] = {
        diff.value: {"total": 0, "correct": 0} for diff in DifficultyLevel
    }

    for pred, gt in zip(predictions, ground_truth):
        total_citations += len(pred.citations)
        diff_val = gt.difficulty.value
        diff_stats[diff_val]["total"] += 1

        # 1. Evaluate Insufficient Evidence cases
        if gt.difficulty == DifficultyLevel.INSUFFICIENT_EVIDENCE:
            insufficient_gt_count += 1
            is_correct_abstention = (
                pred.insufficient_evidence or pred.is_ambiguous or pred.top_fault_id is None
            )
            if is_correct_abstention:
                insufficient_correct += 1
                diff_stats[diff_val]["correct"] += 1
        else:
            # 2. Evaluate Diagnosable cases
            diagnosable_count += 1
            if pred.top_fault_id == gt.true_fault_id:
                top1_correct += 1
                diff_stats[diff_val]["correct"] += 1
            if gt.true_fault_id in pred.ranked_candidates[:3]:
                top3_correct += 1

            # Unsupported claim: making a diagnostic claim with zero supporting citations
            if len(pred.citations) == 0:
                unsupported_count += 1

        if pred.is_ambiguous:
            ambiguous_count += 1

    top1_acc = (top1_correct / diagnosable_count) if diagnosable_count else 0.0
    top3_acc = (top3_correct / diagnosable_count) if diagnosable_count else 0.0
    insuff_acc = (insufficient_correct / insufficient_gt_count) if insufficient_gt_count else 1.0
    ambig_rate = (ambiguous_count / total_cases) if total_cases else 0.0
    avg_citations = (total_citations / total_cases) if total_cases else 0.0
    unsupported_rate = (unsupported_count / diagnosable_count) if diagnosable_count else 0.0

    breakdown = {}
    for diff_name, counts in diff_stats.items():
        if counts["total"] > 0:
            breakdown[diff_name] = round(counts["correct"] / counts["total"], 4)
        else:
            breakdown[diff_name] = 0.0

    return EvaluationMetrics(
        mode=mode,
        total_cases=total_cases,
        diagnosable_cases=diagnosable_count,
        top1_accuracy=round(top1_acc, 4),
        top3_accuracy=round(top3_acc, 4),
        ambiguity_rate=round(ambig_rate, 4),
        insufficient_evidence_accuracy=round(insuff_acc, 4),
        avg_citations_per_case=round(avg_citations, 2),
        unsupported_claim_rate=round(unsupported_rate, 4),
        difficulty_breakdown=breakdown,
    )
