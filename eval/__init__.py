"""
Evaluation and Benchmarking Package for Automotive Digital Twin.
Implements LLM-only, RAG-only, and Full System diagnostic baselines.
"""

from .baselines import (
    BaselinePrediction,
    DiagnosticBaseline,
    LLMOnlyBaseline,
    RAGOnlyBaseline,
    FullSystemBaseline,
)
from .metrics import EvaluationMetrics, compute_metrics

__all__ = [
    "BaselinePrediction",
    "DiagnosticBaseline",
    "LLMOnlyBaseline",
    "RAGOnlyBaseline",
    "FullSystemBaseline",
    "EvaluationMetrics",
    "compute_metrics",
]
