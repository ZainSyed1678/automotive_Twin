"""
LangGraph StateGraph compiler and workflow execution engine.
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END

from model.schema import Observations
from .state import DiagnosticReport, DiagnosticState
from .nodes import (
    intake_node,
    retrieval_node,
    hypothesis_generation_node,
    simulation_scoring_node,
    reporting_node,
)


def create_diagnostic_workflow():
    """
    Constructs and compiles the LangGraph diagnostic workflow.
    """
    workflow = StateGraph(DiagnosticState)

    # 1. Register logical nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("simulation_scoring", simulation_scoring_node)
    workflow.add_node("reporting", reporting_node)

    # 2. Define sequential execution flow
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "retrieval")
    workflow.add_edge("retrieval", "hypothesis_generation")
    workflow.add_edge("hypothesis_generation", "simulation_scoring")
    workflow.add_edge("simulation_scoring", "reporting")
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
) -> DiagnosticReport:
    """
    Executes the end-to-end LangGraph diagnostic pipeline for a given observation set.
    """
    app = get_compiled_app()
    initial_state: DiagnosticState = {
        "case_id": case_id,
        "raw_input": raw_input,
        "observations": observations.model_dump(),
        "citations": [],
    }

    final_state = app.invoke(initial_state)
    report_dict = final_state.get("final_report", {})
    return DiagnosticReport(**report_dict)
