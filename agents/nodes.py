"""
LangGraph workflow nodes for Automotive Digital Twin Diagnosis.
Pure functions implementing Intake, Retrieval, Hypotheses, Simulation/Scoring,
Adaptive Questioning (EIG), Review, and Reporting.
"""

from typing import Any, Dict, List, Optional
from model.engine import FaultModel
from model.schema import Observations
from model.vehicle_registry import vehicle_registry
from rag.store import KnowledgeBase, SimilarCaseIndex
from .eig import select_best_follow_up_question
from .review import review_diagnostic_report
from .state import DiagnosticReport, DiagnosticState


# Shared singleton resources (loaded once in memory)
_fault_model: Optional[FaultModel] = None
_knowledge_base: Optional[KnowledgeBase] = None
_similar_cases: Optional[SimilarCaseIndex] = None


def get_fault_model() -> FaultModel:
    global _fault_model
    if _fault_model is None:
        _fault_model = FaultModel()
    return _fault_model


def get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def get_similar_cases() -> SimilarCaseIndex:
    global _similar_cases
    if _similar_cases is None:
        _similar_cases = SimilarCaseIndex()
    return _similar_cases


# -----------------------------------------------------------------------------
# Node 1: Intake
# -----------------------------------------------------------------------------

def intake_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Parses and normalizes technician observations and diagnostic trouble codes.
    """
    raw_obs = state.get("observations", {})
    dtcs = [c.strip().upper() for c in raw_obs.get("dtc_codes", []) if c.strip()]
    symptoms = {
        k.strip().lower(): v.strip().lower()
        for k, v in raw_obs.get("symptoms", {}).items()
        if k.strip() and v.strip()
    }
    notes = raw_obs.get("notes") or state.get("raw_input")
    vehicle_id = raw_obs.get("vehicle_id") or state.get("vehicle_id") or "generic_petrol"

    normalized_obs = {
        "dtc_codes": sorted(list(set(dtcs))),
        "symptoms": dict(sorted(symptoms.items())),
        "notes": notes,
        "vehicle_id": vehicle_id,
    }

    return {"observations": normalized_obs, "vehicle_id": vehicle_id}


# -----------------------------------------------------------------------------
# Node 2: Retrieval (Hybrid RAG + Similar Cases)
# -----------------------------------------------------------------------------

def retrieval_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Retrieves relevant technical manuals, TSBs, OBD code sheets, and similar historical cases.
    Collects verifiable citation handles.
    """
    kb = get_knowledge_base()
    case_index = get_similar_cases()
    obs_data = state.get("observations", {})
    dtcs = obs_data.get("dtc_codes", [])
    symptoms = obs_data.get("symptoms", {})

    retrieved_docs: List[Dict[str, Any]] = []
    citations: List[str] = list(state.get("citations", []))

    # 1. Direct OBD Code Lookups
    for code in dtcs:
        code_info = kb.get_code_info(code)
        if code_info:
            doc_dict = code_info.model_dump()
            retrieved_docs.append(doc_dict)
            if code_info.citation not in citations:
                citations.append(code_info.citation)

    # 2. Semantic query search over manuals and TSBs
    vehicle_id = obs_data.get("vehicle_id") or state.get("vehicle_id")
    query_parts = list(dtcs) + [f"{k} {v}" for k, v in symptoms.items()]
    query = " ".join(query_parts) if query_parts else "generic engine air fuel diagnosis"
    search_hits = kb.search(query=query, vehicle_id=vehicle_id, top_k=3)
    for hit in search_hits:
        if hit.doc_id not in [d["doc_id"] for d in retrieved_docs]:
            retrieved_docs.append(hit.model_dump())
            if hit.citation not in citations:
                citations.append(hit.citation)

    # 3. Similar Historical Cases Search (strictly dev cases)
    obs_obj = Observations(dtc_codes=dtcs, symptoms=symptoms)
    similar_hits = case_index.find_similar(obs_obj, top_k=3)
    similar_cases = [s.model_dump() for s in similar_hits]
    for sc in similar_cases:
        citation_str = f"[{sc['case_id']}] Confirmed: {sc['confirmed_fault']} (Sim={sc['similarity_score']:.2f})"
        if citation_str not in citations:
            citations.append(citation_str)

    return {
        "retrieved_docs": retrieved_docs,
        "similar_cases": similar_cases,
        "citations": citations,
    }


# -----------------------------------------------------------------------------
# Node 3: Hypothesis Generation
# -----------------------------------------------------------------------------

def hypothesis_generation_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Generates candidate fault hypotheses from observations, retrieved evidence, and model priors.
    """
    model = get_fault_model()
    obs_data = state.get("observations", {})
    dtcs = set(obs_data.get("dtc_codes", []))
    symptoms = set(obs_data.get("symptoms", {}).keys())

    candidates: List[str] = []

    # Faults that explain observed DTCs or symptoms
    for fault in model.list_faults():
        has_dtc_match = any(d in fault.dtcs for d in dtcs)
        has_symptom_match = any(s in fault.symptoms for s in symptoms)
        if has_dtc_match or has_symptom_match:
            candidates.append(fault.fault_id)

    # If evidence is extremely sparse, include all catalogue faults
    if not candidates:
        candidates = [f.fault_id for f in model.list_faults()]

    return {"candidate_hypotheses": candidates}


# -----------------------------------------------------------------------------
# Node 4: Simulation & Scoring
# -----------------------------------------------------------------------------

def simulation_scoring_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Evaluates candidate hypotheses against the causal digital twin using Bayesian likelihood scoring.
    Simulates top candidates to extract predicted behavior and distinguishing tests.
    """
    model = get_fault_model()
    obs_data = state.get("observations", {})
    obs_obj = Observations(
        dtc_codes=obs_data.get("dtc_codes", []),
        symptoms=obs_data.get("symptoms", {}),
        notes=obs_data.get("notes"),
        vehicle_id=obs_data.get("vehicle_id", "generic_petrol"),
    )

    diag_result = model.score_hypotheses(obs_obj)

    # Simulate top 3 candidate faults
    sim_dict: Dict[str, Dict[str, Any]] = {}
    for hyp in diag_result.ranked_hypotheses[:3]:
        sim = model.simulate_fault(hyp.fault_id, vehicle_id=obs_obj.vehicle_id)
        sim_dict[hyp.fault_id] = sim.model_dump()

    return {
        "diagnostic_result": diag_result.model_dump(),
        "simulation_results": sim_dict,
    }


# -----------------------------------------------------------------------------
# Node 5: Adaptive Question Selection (Expected Information Gain)
# -----------------------------------------------------------------------------

def question_selection_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Selects the optimal follow-up diagnostic test using Expected Information Gain (EIG)
    to actively reduce uncertainty between leading hypotheses.
    """
    model = get_fault_model()
    diag_data = state.get("diagnostic_result") or {}
    obs_data = state.get("observations", {})
    ranked_hyps = diag_data.get("ranked_hypotheses") or []
    confidence_gap = diag_data.get("confidence_gap", 0.0)
    is_ambiguous = diag_data.get("is_ambiguous", False)
    insufficient_evidence = diag_data.get("insufficient_evidence", False)

    best_question = select_best_follow_up_question(
        ranked_hypotheses=ranked_hyps,
        current_observations=obs_data,
        fault_model=model,
        confidence_gap=confidence_gap,
        is_ambiguous=is_ambiguous,
        insufficient_evidence=insufficient_evidence,
        min_eig_threshold=0.04,
    )

    question_dict = best_question.model_dump() if best_question else None
    return {"follow_up_question": question_dict}


# -----------------------------------------------------------------------------
# Node 6: Review (Self-Review / Verification Gate)
# -----------------------------------------------------------------------------

def review_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Verification gate that validates evidence grounding, checks subsystem boundaries,
    and enforces epistemic modesty on ambiguous or insufficient evidence cases.
    """
    model = get_fault_model()
    review_res = review_diagnostic_report(state, model)
    return {"review_result": review_res.model_dump()}


# -----------------------------------------------------------------------------
# Node 7: Reporting
# -----------------------------------------------------------------------------

def reporting_node(state: DiagnosticState) -> Dict[str, Any]:
    """
    Compiles evidence, citations, simulation predictions, review verification,
    and adaptive question recommendations into an actionable report.
    """
    diag_data = state.get("diagnostic_result") or {}
    sim_dict = state.get("simulation_results") or {}
    citations = state.get("citations") or []
    ranked_hyps = diag_data.get("ranked_hypotheses") or []
    follow_up_q = state.get("follow_up_question")
    review_data = state.get("review_result") or {}

    top_hyp = ranked_hyps[0] if ranked_hyps else {}
    top_fault_id = top_hyp.get("fault_id")
    top_fault_name = top_hyp.get("fault_name")
    top_component = top_hyp.get("component_id")
    confidence_score = top_hyp.get("posterior_prob", 0.0)
    confidence_gap = diag_data.get("confidence_gap", 0.0)
    is_ambiguous = diag_data.get("is_ambiguous", False)
    insufficient_evidence = diag_data.get("insufficient_evidence", False)
    suggested_tests = diag_data.get("suggested_tests") or []

    supporting: List[str] = top_hyp.get("evidence_matches", [])
    contradictions: List[str] = top_hyp.get("evidence_contradictions", [])

    # Compose executive summary
    if review_data.get("adjusted_summary"):
        summary = review_data["adjusted_summary"]
    elif insufficient_evidence:
        summary = (
            "INSUFFICIENT EVIDENCE: The reported observations are too sparse or non-specific "
            "to reliably isolate a single root cause. Follow-up physical inspection is required."
        )
    elif is_ambiguous:
        runner_up = ranked_hyps[1].get("fault_name") if len(ranked_hyps) > 1 else "alternative fault"
        q_hint = f" Recommended next test: {follow_up_q['test_name']}." if follow_up_q else ""
        summary = (
            f"AMBIGUOUS DIAGNOSIS: Primary candidate is '{top_fault_name}' ({confidence_score * 100:.1f}%), "
            f"closely competed by '{runner_up}' (margin: {confidence_gap * 100:.1f}%). "
            f"Distinguishing tests must be performed before component replacement.{q_hint}"
        )
    else:
        summary = (
            f"HIGH CONFIDENCE DIAGNOSIS: Leading hypothesis is '{top_fault_name}' "
            f"({confidence_score * 100:.1f}% confidence, margin: {confidence_gap * 100:.1f}%). "
            f"Observed pattern aligns with digital twin causal simulation."
        )

    vehicle_id = (state.get("observations") or {}).get("vehicle_id") or state.get("vehicle_id") or "generic_petrol"
    vehicle_prof = vehicle_registry.get_profile(vehicle_id)
    vehicle_name = vehicle_prof.name if vehicle_prof else vehicle_id

    report = DiagnosticReport(
        case_id=state.get("case_id"),
        vehicle_id=vehicle_id,
        vehicle_name=vehicle_name,
        top_fault_id=top_fault_id,
        top_fault_name=top_fault_name,
        component_id=top_component,
        confidence_score=confidence_score,
        confidence_gap=confidence_gap,
        is_ambiguous=is_ambiguous,
        insufficient_evidence=insufficient_evidence,
        ranked_hypotheses=ranked_hyps[:5],
        supporting_evidence=supporting,
        contradictory_evidence=contradictions,
        recommended_tests=suggested_tests[:4],
        citations=citations,
        follow_up_question=follow_up_q,
        review_status=review_data.get("review_status", "PASSED"),
        review_notes=review_data.get("review_notes", []),
        executive_summary=summary,
    )

    return {"final_report": report.model_dump()}
