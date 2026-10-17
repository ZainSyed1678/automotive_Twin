"""
Evaluation metrics computation for diagnostic baselines.
Measures Top-1, Top-3, ambiguity detection, insufficient evidence handling, and citation rate.
"""

from typing import List
from pydantic import BaseModel

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

    for pred, gt in zip(predictions, ground_truth):
        total_citations += len(pred.citations)

        # 1. Evaluate Insufficient Evidence cases
        if gt.difficulty == DifficultyLevel.INSUFFICIENT_EVIDENCE:
            insufficient_gt_count += 1
            if pred.insufficient_evidence or pred.is_ambiguous or pred.top_fault_id is None:
                insufficient_correct += 1
        else:
            # 2. Evaluate Diagnosable cases
            diagnosable_count += 1
            if pred.top_fault_id == gt.true_fault_id:
                top1_correct += 1
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
    )
