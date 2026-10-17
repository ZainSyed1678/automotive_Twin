"""
Automated test suite for M4: LangGraph Diagnostic Workflow, Baselines, and Evaluation Metrics.
"""

import pytest
from agents import run_diagnosis, create_diagnostic_workflow, DiagnosticReport
from casegen.schema import AgentCase, DifficultyLevel, GroundTruthCase
from eval import (
    BaselinePrediction,
    LLMOnlyBaseline,
    RAGOnlyBaseline,
    FullSystemBaseline,
    compute_metrics,
)
from model.schema import Observations


# -----------------------------------------------------------------------------
# 1. LangGraph Workflow Tests
# -----------------------------------------------------------------------------

def test_workflow_compilation():
    """Verify that the LangGraph StateGraph compiles cleanly."""
    app = create_diagnostic_workflow()
    assert app is not None


def test_end_to_end_diagnosis():
    """Verify end-to-end diagnostic workflow on a resolved vacuum leak case."""
    obs = Observations(
        dtc_codes=["P0171"],
        symptoms={
            "fuel_pressure": "normal",
            "fuel_trim_at_high_rpm": "normalizes",
            "audible_hiss": "present",
        }
    )
    report = run_diagnosis(obs, case_id="CASE-TEST-001")

    assert isinstance(report, DiagnosticReport)
    assert report.top_fault_id == "intake_vacuum_leak"
    assert report.confidence_score > 0.85
    assert report.is_ambiguous is False
    assert len(report.citations) >= 2
    assert any("DOC-" in c for c in report.citations)
    assert len(report.recommended_tests) >= 1
    assert "HIGH CONFIDENCE DIAGNOSIS" in report.executive_summary


def test_insufficient_evidence_workflow():
    """Verify workflow correctly handles and reports insufficient observations."""
    obs = Observations(
        dtc_codes=[],
        symptoms={"visual_inspection": "clean"}
    )
    report = run_diagnosis(obs, case_id="CASE-TEST-002")

    assert report.insufficient_evidence is True or report.is_ambiguous is True
    assert "INSUFFICIENT EVIDENCE" in report.executive_summary or "AMBIGUOUS" in report.executive_summary


# -----------------------------------------------------------------------------
# 2. Baseline Comparison Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_case():
    obs = Observations(
        dtc_codes=["P0171", "P0087"],
        symptoms={"fuel_pressure": "low", "fuel_trim_at_high_rpm": "worsens"}
    )
    return AgentCase(case_id="CASE-TEST-003", observations=obs)


def test_llm_only_baseline(sample_case: AgentCase):
    """Verify LLM-only baseline produces heuristic prediction without citations."""
    runner = LLMOnlyBaseline()
    pred = runner.diagnose(sample_case)

    assert isinstance(pred, BaselinePrediction)
    assert pred.mode == "LLM-Only"
    assert pred.top_fault_id in ["fuel_pump_weak", "intake_vacuum_leak"]
    assert len(pred.citations) == 0, "LLM-only baseline must have zero citations"


def test_rag_only_baseline(sample_case: AgentCase):
    """Verify RAG-only baseline produces document-backed prediction."""
    runner = RAGOnlyBaseline()
    pred = runner.diagnose(sample_case)

    assert isinstance(pred, BaselinePrediction)
    assert pred.mode == "RAG-Only"
    assert pred.top_fault_id is not None
    assert len(pred.citations) > 0, "RAG-only baseline must include citations"


def test_full_system_baseline(sample_case: AgentCase):
    """Verify Full-System baseline integrates simulation, scoring, and citations."""
    runner = FullSystemBaseline()
    pred = runner.diagnose(sample_case)

    assert isinstance(pred, BaselinePrediction)
    assert pred.mode == "Full-System"
    assert pred.top_fault_id == "fuel_pump_weak"
    assert pred.confidence_score > 0.80
    assert len(pred.citations) > 0


# -----------------------------------------------------------------------------
# 3. Metrics Computation Tests
# -----------------------------------------------------------------------------

def test_compute_metrics():
    """Verify metric calculations for diagnosable and insufficient cases."""
    gt_cases = [
        GroundTruthCase(
            case_id="CASE-001",
            true_fault_id="intake_vacuum_leak",
            difficulty=DifficultyLevel.STRAIGHTFORWARD,
            observations=Observations(dtc_codes=["P0171"]),
        ),
        GroundTruthCase(
            case_id="CASE-002",
            true_fault_id="fuel_pump_weak",
            difficulty=DifficultyLevel.STRAIGHTFORWARD,
            observations=Observations(dtc_codes=["P0087"]),
        ),
        GroundTruthCase(
            case_id="CASE-003",
            true_fault_id="intake_vacuum_leak",
            difficulty=DifficultyLevel.INSUFFICIENT_EVIDENCE,
            observations=Observations(symptoms={"idle_quality": "rough"}),
        ),
    ]

    preds = [
        BaselinePrediction(
            case_id="CASE-001",
            top_fault_id="intake_vacuum_leak",
            ranked_candidates=["intake_vacuum_leak"],
            citations=["[DOC-1]"],
            mode="Test",
        ),
        BaselinePrediction(
            case_id="CASE-002",
            top_fault_id="fuel_injector_clogged",  # Incorrect top-1, but in top-3
            ranked_candidates=["fuel_injector_clogged", "fuel_pump_weak"],
            citations=["[DOC-2]"],
            mode="Test",
        ),
        BaselinePrediction(
            case_id="CASE-003",
            top_fault_id=None,
            insufficient_evidence=True,
            mode="Test",
        ),
    ]

    metrics = compute_metrics(preds, gt_cases)
    assert metrics.total_cases == 3
    assert metrics.diagnosable_cases == 2
    assert metrics.top1_accuracy == 0.50
    assert metrics.top3_accuracy == 1.00
    assert metrics.insufficient_evidence_accuracy == 1.00
    assert metrics.unsupported_claim_rate == 0.0
