"""
Implementation of the three diagnostic baselines:
1. LLM-Only: Heuristic reasoning without retrieval or digital twin simulation.
2. RAG-Only: Technical retrieval and similar-case matching without causal simulation.
3. Full System: Retrieval + similar cases + digital twin causal simulation & Bayesian scoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agents.workflow import run_diagnosis
from casegen.schema import AgentCase
from model.engine import FaultModel
from model.schema import Observations
from rag.store import KnowledgeBase, SimilarCaseIndex


class BaselinePrediction(BaseModel):
    case_id: str
    top_fault_id: Optional[str] = None
    ranked_candidates: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_gap: float = 0.0
    is_ambiguous: bool = False
    insufficient_evidence: bool = False
    citations: List[str] = Field(default_factory=list)
    mode: str = "unknown"


class DiagnosticBaseline(ABC):
    @abstractmethod
    def diagnose(self, case: AgentCase) -> BaselinePrediction:
        pass


# -----------------------------------------------------------------------------
# Mode 1: LLM-Only Baseline (Zero RAG, Zero Digital Twin)
# -----------------------------------------------------------------------------

class LLMOnlyBaseline(DiagnosticBaseline):
    """
    Simulates direct LLM diagnostic reasoning without RAG retrieval or digital twin simulation.
    Maps DTCs and primary symptoms directly using static zero-shot association heuristics.
    """

    DIRECT_DTC_MAP = {
        "P0101": "maf_contamination",
        "P0102": "wiring_open_circuit",
        "P0103": "wiring_short_circuit",
        "P0106": "map_sensor_drift",
        "P0107": "map_sensor_drift",
        "P0117": "coolant_sensor_stuck_low",
        "P0121": "throttle_sensor_drift",
        "P0133": "o2_sensor_slow_response",
        "P0087": "fuel_pump_weak",
        "P0171": "intake_vacuum_leak",  # Common naive assumption
        "P0507": "intake_vacuum_leak",
        "P0300": "fuel_injector_clogged",
        "P0301": "fuel_injector_clogged",
        "P0100": "connector_corrosion",
        "P0110": "connector_corrosion",
        "P0120": "connector_corrosion",
    }

    def diagnose(self, case: AgentCase) -> BaselinePrediction:
        dtcs = case.observations.dtc_codes
        symptoms = case.observations.symptoms

        num_obs = len(dtcs) + len(symptoms)
        if num_obs == 0 or (len(dtcs) == 0 and len(symptoms) <= 1):
            return BaselinePrediction(
                case_id=case.case_id,
                top_fault_id=None,
                ranked_candidates=[],
                confidence_score=0.10,
                confidence_gap=0.0,
                is_ambiguous=True,
                insufficient_evidence=True,
                citations=[],
                mode="LLM-Only",
            )

        # Primary DTC match
        candidates: List[str] = []
        for code in dtcs:
            if code in self.DIRECT_DTC_MAP:
                cand = self.DIRECT_DTC_MAP[code]
                if cand not in candidates:
                    candidates.append(cand)

        # Symptom-based fallbacks
        if not candidates:
            if "fuel_pressure" in symptoms and symptoms["fuel_pressure"] == "low":
                candidates.append("fuel_pump_weak")
            elif "audible_hiss" in symptoms and symptoms["audible_hiss"] == "present":
                candidates.append("intake_vacuum_leak")
            elif "tps_voltage_sweep" in symptoms:
                candidates.append("throttle_sensor_drift")
            elif "live_coolant_temp" in symptoms:
                candidates.append("coolant_sensor_stuck_low")
            else:
                candidates.append("intake_vacuum_leak")

        top_fault = candidates[0] if candidates else "intake_vacuum_leak"
        is_ambiguous = len(candidates) > 1 or dtcs == ["P0171"]

        return BaselinePrediction(
            case_id=case.case_id,
            top_fault_id=top_fault,
            ranked_candidates=candidates,
            confidence_score=0.60 if not is_ambiguous else 0.35,
            confidence_gap=0.20 if not is_ambiguous else 0.05,
            is_ambiguous=is_ambiguous,
            insufficient_evidence=False,
            citations=[],  # Zero citations in LLM-only mode
            mode="LLM-Only",
        )


# -----------------------------------------------------------------------------
# Mode 2: RAG-Only Baseline (RAG + Similar Cases, Zero Digital Twin Physics)
# -----------------------------------------------------------------------------

class RAGOnlyBaseline(DiagnosticBaseline):
    """
    Ranks hypotheses using retrieved technical documentation and historical similar cases,
    without causal digital twin simulation or likelihood scoring.
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None, cases: Optional[SimilarCaseIndex] = None):
        self.kb = kb or KnowledgeBase()
        self.similar_cases = cases or SimilarCaseIndex()
        self.model = FaultModel()

    def diagnose(self, case: AgentCase) -> BaselinePrediction:
        dtcs = case.observations.dtc_codes
        symptoms = case.observations.symptoms

        if len(dtcs) == 0 and len(symptoms) <= 1:
            return BaselinePrediction(
                case_id=case.case_id,
                top_fault_id=None,
                ranked_candidates=[],
                confidence_score=0.10,
                confidence_gap=0.0,
                is_ambiguous=True,
                insufficient_evidence=True,
                citations=[],
                mode="RAG-Only",
            )

        citations: List[str] = []
        candidate_scores: Dict[str, float] = {f.fault_id: 0.05 for f in self.model.list_faults()}

        # 1. Evidence from technical documents
        query = " ".join(list(dtcs) + [f"{k} {v}" for k, v in symptoms.items()])
        docs = self.kb.search(query=query, top_k=3)
        for doc in docs:
            citations.append(doc.citation)
            doc_text = f"{doc.title} {doc.snippet}".lower()
            for fault_id in candidate_scores:
                if fault_id in doc_text or (doc.component_id and doc.component_id in fault_id):
                    candidate_scores[fault_id] += doc.score * 0.40

        # 2. Evidence from similar historical cases (development set only)
        similar = self.similar_cases.find_similar(case.observations, top_k=3)
        for sc in similar:
            citations.append(f"[{sc.case_id}] Confirmed: {sc.confirmed_fault}")
            if sc.confirmed_fault in candidate_scores:
                candidate_scores[sc.confirmed_fault] += sc.similarity_score * 0.60

        # Rank candidates
        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_cand, top_score = ranked[0]
        runner_up_cand, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        # Normalize heuristic score
        total_score = sum(s for _, s in ranked) or 1.0
        prob = round(top_score / total_score, 4)
        gap = round((top_score - runner_up_score) / total_score, 4)

        return BaselinePrediction(
            case_id=case.case_id,
            top_fault_id=top_cand,
            ranked_candidates=[f for f, _ in ranked[:5]],
            confidence_score=prob,
            confidence_gap=gap,
            is_ambiguous=gap < 0.15,
            insufficient_evidence=False,
            citations=citations,
            mode="RAG-Only",
        )


# -----------------------------------------------------------------------------
# Mode 3: Full System Baseline (LangGraph Agent: RAG + Digital Twin Simulation)
# -----------------------------------------------------------------------------

class FullSystemBaseline(DiagnosticBaseline):
    """
    Executes the full LangGraph diagnostic workflow:
    Intake -> Retrieval -> Hypotheses -> Digital Twin Causal Simulation -> Scoring -> Reporting.
    """

    def diagnose(self, case: AgentCase) -> BaselinePrediction:
        report = run_diagnosis(case.observations, case_id=case.case_id)

        ranked = [h.get("fault_id") for h in report.ranked_hypotheses if h.get("fault_id")]
        return BaselinePrediction(
            case_id=case.case_id,
            top_fault_id=report.top_fault_id,
            ranked_candidates=ranked,
            confidence_score=report.confidence_score,
            confidence_gap=report.confidence_gap,
            is_ambiguous=report.is_ambiguous,
            insufficient_evidence=report.insufficient_evidence,
            citations=report.citations,
            mode="Full-System",
        )
