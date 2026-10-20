"""
LangGraph StateGraph compiler and workflow execution engine.
Integrates Intake, Retrieval, Hypotheses, Simulation/Scoring, Adaptive Questioning (EIG),
Review Verification, and Reporting. Supports single-shot and interactive multi-turn diagnosis.
"""

from typing import Any, Callable, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from model.schema import Observations
from .state import DiagnosticReport, DiagnosticState
from .nodes import (
    intake_node,
    retrieval_node,
    hypothesis_generation_node,
    simulation_scoring_node,
    question_selection_node,
    review_node,
    reporting_node,
)


def create_diagnostic_workflow():
    """
    Constructs and compiles the full LangGraph diagnostic workflow.
    Workflow topology:
    START -> intake -> retrieval -> hypothesis_generation -> simulation_scoring
          -> question_selection -> review -> reporting -> END
    """
    workflow = StateGraph(DiagnosticState)

    # 1. Register logical nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("simulation_scoring", simulation_scoring_node)
    workflow.add_node("question_selection", question_selection_node)
    workflow.add_node("review", review_node)
    workflow.add_node("reporting", reporting_node)

    # 2. Define sequential execution flow
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "retrieval")
    workflow.add_edge("retrieval", "hypothesis_generation")
    workflow.add_edge("hypothesis_generation", "simulation_scoring")
    workflow.add_edge("simulation_scoring", "question_selection")
    workflow.add_edge("question_selection", "review")
    workflow.add_edge("review", "reporting")
    workflow.add_edge("reporting", END)

    return workflow.compile()


# Singleton compiled graph runner
_compiled_app = None


def get_compiled_app():
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = create_diagnostic_workflow()
    return _compiled_app


def run_diagnosis(
    observations: Observations,
    case_id: Optional[str] = None,
    raw_input: Optional[str] = None,
    turn_count: int = 1,
) -> DiagnosticReport:
    """
    Executes a single pass of the LangGraph diagnostic pipeline.
    Produces an evidence-backed report with EIG follow-up question and review verification.
    """
    app = get_compiled_app()
    initial_state: DiagnosticState = {
        "case_id": case_id,
        "vehicle_id": getattr(observations, "vehicle_id", "generic_petrol"),
        "raw_input": raw_input,
        "observations": observations.model_dump(),
        "citations": [],
        "turn_count": turn_count,
    }

    final_state = app.invoke(initial_state)
    report_dict = final_state.get("final_report", {})
    return DiagnosticReport(**report_dict)


def run_interactive_diagnosis(
    initial_observations: Observations,
    answer_callback: Callable[[Dict[str, Any]], Optional[str]],
    max_turns: int = 3,
    case_id: Optional[str] = None,
) -> List[DiagnosticReport]:
    """
    Executes an interactive multi-turn diagnostic loop driven by Expected Information Gain.
    In each turn:
    1. Runs diagnosis to rank hypotheses and select the highest-EIG question.
    2. Calls answer_callback(question_dict) to simulate technician feedback.
    3. If an answer is provided, updates observations and repeats until:
       - Confidence threshold reached (is_ambiguous is False and confidence_gap >= 0.25)
       - No further question is recommended (EIG < threshold)
       - max_turns is exhausted or callback returns None.
    Returns the sequence of DiagnosticReport objects across all turns.
    """
    history: List[DiagnosticReport] = []
    current_obs = Observations(
        dtc_codes=list(initial_observations.dtc_codes),
        symptoms=dict(initial_observations.symptoms),
        notes=initial_observations.notes,
    )

    for turn in range(1, max_turns + 1):
        report = run_diagnosis(current_obs, case_id=case_id, turn_count=turn)
        history.append(report)

        # Stopping criterion: confident diagnosis reached or no question selected
        if not report.is_ambiguous and report.confidence_gap >= 0.25 and not report.insufficient_evidence:
            break

        question = report.follow_up_question
        if not question:
            break

        # Query the technician (or callback)
        answer = answer_callback(question)
        if not answer:
            break

        # Record answer into symptoms for next turn
        test_id = question["test_id"]
        current_obs.symptoms[test_id] = str(answer).strip().lower()

    return history
