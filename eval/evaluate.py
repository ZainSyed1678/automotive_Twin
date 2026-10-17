"""
Benchmark evaluation runner for M4.
Executes LLM-Only, RAG-Only, and Full-System baselines on the development dataset.
"""

import json
from pathlib import Path
from typing import List

from casegen.schema import AgentCase, GroundTruthCase
from .baselines import LLMOnlyBaseline, RAGOnlyBaseline, FullSystemBaseline
from .metrics import EvaluationMetrics, compute_metrics


def run_benchmark(
    dev_cases_path: str = "data/cases_dev.json",
    output_path: str = "data/baseline_evaluation_m4.json",
) -> List[EvaluationMetrics]:
    path = Path(dev_cases_path)
    if not path.exists():
        raise FileNotFoundError(f"Development cases file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    gt_cases = [GroundTruthCase(**item) for item in raw_data]
    agent_cases = [case.to_agent_case() for case in gt_cases]

    print("=" * 80)
    print(f"M4 Benchmark Evaluation: Evaluating {len(gt_cases)} Development Cases")
    print("=" * 80)

    baselines = [
        ("LLM-Only", LLMOnlyBaseline()),
        ("RAG-Only", RAGOnlyBaseline()),
        ("Full-System", FullSystemBaseline()),
    ]

    all_metrics: List[EvaluationMetrics] = []

    for name, runner in baselines:
        print(f"\nRunning {name} baseline...")
        predictions = []
        for i, case in enumerate(agent_cases, start=1):
            pred = runner.diagnose(case)
            predictions.append(pred)
            if i % 35 == 0 or i == len(agent_cases):
                print(f"  Processed {i}/{len(agent_cases)} cases...")

        metrics = compute_metrics(predictions, gt_cases)
        all_metrics.append(metrics)

    # Print Comparative Table
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION RESULTS (140 Development Cases)")
    print("=" * 80)
    print(
        f"{'Metric':<32} | {'LLM-Only':<12} | {'RAG-Only':<12} | {'Full System':<12}"
    )
    print("-" * 80)

    m_llm, m_rag, m_full = all_metrics

    print(f"{'Top-1 Diagnostic Accuracy':<32} | {m_llm.top1_accuracy * 100:>10.1f}% | {m_rag.top1_accuracy * 100:>10.1f}% | {m_full.top1_accuracy * 100:>10.1f}%")
    print(f"{'Top-3 Diagnostic Accuracy':<32} | {m_llm.top3_accuracy * 100:>10.1f}% | {m_rag.top3_accuracy * 100:>10.1f}% | {m_full.top3_accuracy * 100:>10.1f}%")
    print(f"{'Insufficient Evidence Accuracy':<32} | {m_llm.insufficient_evidence_accuracy * 100:>10.1f}% | {m_rag.insufficient_evidence_accuracy * 100:>10.1f}% | {m_full.insufficient_evidence_accuracy * 100:>10.1f}%")
    print(f"{'Ambiguity Detection Rate':<32} | {m_llm.ambiguity_rate * 100:>10.1f}% | {m_rag.ambiguity_rate * 100:>10.1f}% | {m_full.ambiguity_rate * 100:>10.1f}%")
    print(f"{'Avg Citations per Case':<32} | {m_llm.avg_citations_per_case:>11.2f} | {m_rag.avg_citations_per_case:>11.2f} | {m_full.avg_citations_per_case:>11.2f}")
    print(f"{'Unsupported Claim Rate':<32} | {m_llm.unsupported_claim_rate * 100:>10.1f}% | {m_rag.unsupported_claim_rate * 100:>10.1f}% | {m_full.unsupported_claim_rate * 100:>10.1f}%")
    print("=" * 80)

    # Save results to disk
    out = Path(output_path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in all_metrics], f, indent=2)
    print(f"Saved evaluation metrics to: {out}\n")

    return all_metrics


if __name__ == "__main__":
    run_benchmark()
