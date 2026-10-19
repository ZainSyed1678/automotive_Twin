"""
Milestone 7: Comprehensive Held-Out Benchmark & Uncertainty Calibration Runner.
Evaluates LLM-Only, RAG-Only, and Full-System on the 60 held-out evaluation cases
(strictly isolated with 0 observation fingerprint collisions).
Computes Brier scores, Expected Calibration Error (ECE), and multi-turn adaptive gains.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from agents.workflow import run_diagnosis
from casegen.schema import AgentCase, DifficultyLevel, GroundTruthCase
from model.engine import FaultModel
from .baselines import FullSystemBaseline, LLMOnlyBaseline, RAGOnlyBaseline
from .calibration import compute_calibration_metrics
from .metrics import EvaluationMetrics, compute_metrics


def evaluate_adaptive_questioning_on_heldout(
    ambiguous_cases: List[GroundTruthCase],
    fault_model: FaultModel,
    full_system: FullSystemBaseline,
) -> Dict[str, Any]:
    """
    Evaluates how effectively EIG adaptive questioning resolves ambiguous held-out cases
    when simulated technician feedback is provided based on the true underlying fault.
    """
    if not ambiguous_cases:
        return {"ambiguous_eval_count": 0, "resolved_count": 0, "avg_questions": 0.0}

    initial_correct = 0
    resolved_correct = 0
    total_turns = 0

    for gt in ambiguous_cases:
        agent_case = gt.to_agent_case()
        # Turn 1
        initial_pred = full_system.diagnose(agent_case)
        if initial_pred.top_fault_id == gt.true_fault_id:
            initial_correct += 1

        # Simulate interactive turns using true fault expected symptoms
        true_fault = fault_model.get_fault(gt.true_fault_id)
        current_obs = agent_case.observations.model_copy(deep=True)
        turn = 1

        while turn < 3:
            current_case = AgentCase(case_id=agent_case.case_id, observations=current_obs)
            # Check if current diagnosis is confident
            pred = full_system.diagnose(current_case)
            if not pred.is_ambiguous and pred.confidence_gap >= 0.25:
                break

            # Find recommended test
            diag_report = run_diagnosis(current_obs, case_id=agent_case.case_id)
            q = diag_report.follow_up_question
            if not q:
                break

            test_id = q["test_id"]
            # Look up ground-truth expected symptom state for this fault
            if true_fault and test_id in true_fault.symptoms:
                states = true_fault.symptoms[test_id]
                best_state = max(states.items(), key=lambda x: x[1])[0]
            else:
                best_state = "normal"

            current_obs.symptoms[test_id] = best_state
            turn += 1

        final_case = AgentCase(case_id=agent_case.case_id, observations=current_obs)
        final_pred = full_system.diagnose(final_case)
        if final_pred.top_fault_id == gt.true_fault_id:
            resolved_correct += 1
        total_turns += turn

    n_cases = len(ambiguous_cases)
    return {
        "ambiguous_eval_count": n_cases,
        "initial_top1_accuracy": round(initial_correct / n_cases, 4),
        "post_questioning_top1_accuracy": round(resolved_correct / n_cases, 4),
        "avg_turns_to_resolve": round(total_turns / n_cases, 2),
        "accuracy_gain_pct": round(((resolved_correct - initial_correct) / n_cases) * 100.0, 1),
    }


def run_heldout_evaluation(
    eval_cases_path: str = "data/cases_eval.json",
    output_path: str = "data/final_evaluation_m7.json",
) -> Dict[str, Any]:
    """
    Runs the complete Milestone 7 benchmark on held-out evaluation cases.
    """
    path = Path(eval_cases_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation cases not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    gt_cases = [GroundTruthCase(**item) for item in raw_data]
    agent_cases = [case.to_agent_case() for case in gt_cases]

    print("=" * 80)
    print(f"MILESTONE 7: HELD-OUT EVALUATION BENCHMARK ({len(gt_cases)} CASES)")
    print("=" * 80)

    fault_model = FaultModel()
    baselines = [
        ("LLM-Only", LLMOnlyBaseline()),
        ("RAG-Only", RAGOnlyBaseline()),
        ("Full-System", FullSystemBaseline()),
    ]

    all_metrics: List[EvaluationMetrics] = []
    calibration_reports: Dict[str, Any] = {}

    for name, runner in baselines:
        print(f"\nEvaluating {name} on {len(agent_cases)} held-out cases...")
        predictions = []
        for i, case in enumerate(agent_cases, start=1):
            pred = runner.diagnose(case)
            predictions.append(pred)

        # 1. Performance Metrics
        metrics = compute_metrics(predictions, gt_cases)

        # 2. Uncertainty Calibration Metrics (Brier Score, ECE)
        cal_report = compute_calibration_metrics(predictions, gt_cases, num_bins=5)
        metrics.brier_score = cal_report.brier_score
        metrics.expected_calibration_error = cal_report.expected_calibration_error
        calibration_reports[name] = cal_report.model_dump()

        all_metrics.append(metrics)

    # 3. Evaluate Adaptive Questioning on Ambiguous Eval Cases
    ambiguous_eval_cases = [c for c in gt_cases if c.difficulty == DifficultyLevel.AMBIGUOUS]
    adaptive_gains = evaluate_adaptive_questioning_on_heldout(
        ambiguous_eval_cases, fault_model, FullSystemBaseline()
    )

    # Print Formatted Milestone 7 Comparative Summary
    print("\n" + "=" * 80)
    print(f"{'METRIC':<36} | {'LLM-ONLY':<11} | {'RAG-ONLY':<11} | {'FULL SYSTEM':<11}")
    print("=" * 80)

    m_llm, m_rag, m_full = all_metrics

    print(f"{'Top-1 Diagnostic Accuracy':<36} | {m_llm.top1_accuracy * 100:>9.1f}% | {m_rag.top1_accuracy * 100:>9.1f}% | {m_full.top1_accuracy * 100:>9.1f}%")
    print(f"{'Top-3 Diagnostic Accuracy':<36} | {m_llm.top3_accuracy * 100:>9.1f}% | {m_rag.top3_accuracy * 100:>9.1f}% | {m_full.top3_accuracy * 100:>9.1f}%")
    print(f"{'Insufficient Evidence Handling':<36} | {m_llm.insufficient_evidence_accuracy * 100:>9.1f}% | {m_rag.insufficient_evidence_accuracy * 100:>9.1f}% | {m_full.insufficient_evidence_accuracy * 100:>9.1f}%")
    print(f"{'Ambiguity Detection Rate':<36} | {m_llm.ambiguity_rate * 100:>9.1f}% | {m_rag.ambiguity_rate * 100:>9.1f}% | {m_full.ambiguity_rate * 100:>9.1f}%")
    print(f"{'Expected Calibration Error (ECE)':<36} | {m_llm.expected_calibration_error:>10.3f} | {m_rag.expected_calibration_error:>10.3f} | {m_full.expected_calibration_error:>10.3f}")
    print(f"{'Brier Score (lower is better)':<36} | {m_llm.brier_score:>10.3f} | {m_rag.brier_score:>10.3f} | {m_full.brier_score:>10.3f}")
    print(f"{'Avg Citations per Case':<36} | {m_llm.avg_citations_per_case:>10.2f} | {m_rag.avg_citations_per_case:>10.2f} | {m_full.avg_citations_per_case:>10.2f}")
    print(f"{'Unsupported Claim Rate':<36} | {m_llm.unsupported_claim_rate * 100:>9.1f}% | {m_rag.unsupported_claim_rate * 100:>9.1f}% | {m_full.unsupported_claim_rate * 100:>9.1f}%")
    print("=" * 80)

    # Print Stratified Difficulty Breakdown
    print("\n" + "=" * 80)
    print("STRATIFIED ACCURACY BY DIFFICULTY TIER (HELD-OUT EVALUATION)")
    print("=" * 80)
    print(f"{'Difficulty Tier':<28} | {'LLM-Only':<11} | {'RAG-Only':<11} | {'Full System':<11}")
    print("-" * 80)
    for tier in DifficultyLevel:
        name_t = tier.value.replace("_", " ").title()
        acc_llm = m_llm.difficulty_breakdown.get(tier.value, 0.0) * 100
        acc_rag = m_rag.difficulty_breakdown.get(tier.value, 0.0) * 100
        acc_full = m_full.difficulty_breakdown.get(tier.value, 0.0) * 100
        print(f"{name_t:<28} | {acc_llm:>9.1f}% | {acc_rag:>9.1f}% | {acc_full:>9.1f}%")
    print("=" * 80)

    # Print Adaptive Questioning Disambiguation Results
    print("\n" + "=" * 80)
    print("ADAPTIVE QUESTIONING MULTI-TURN DISAMBIGUATION (EVALUATION SET)")
    print("=" * 80)
    print(f"Ambiguous Cases Evaluated:       {adaptive_gains['ambiguous_eval_count']}")
    print(f"Initial Top-1 Accuracy:          {adaptive_gains['initial_top1_accuracy'] * 100:.1f}%")
    print(f"Post-Questioning Top-1 Accuracy: {adaptive_gains['post_questioning_top1_accuracy'] * 100:.1f}%")
    print(f"Net Accuracy Gain:               +{adaptive_gains['accuracy_gain_pct']:.1f}%")
    print(f"Average Questions to Resolve:    {adaptive_gains['avg_turns_to_resolve'] - 1:.2f}")
    print("=" * 80 + "\n")

    final_results = {
        "benchmark_metadata": {
            "eval_cases_path": eval_cases_path,
            "total_eval_cases": len(gt_cases),
            "diagnosable_eval_cases": len(gt_cases) - len([c for c in gt_cases if c.difficulty == DifficultyLevel.INSUFFICIENT_EVIDENCE]),
        },
        "comparative_metrics": [m.model_dump() for m in all_metrics],
        "calibration_reports": calibration_reports,
        "adaptive_questioning_gains": adaptive_gains,
    }

    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"Saved full final benchmark results to: {out_file}\n")
    return final_results


if __name__ == "__main__":
    run_heldout_evaluation()
