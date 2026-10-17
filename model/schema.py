"""
Data models and schemas for Automotive Digital Twin Fault Diagnosis.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Component(BaseModel):
    id: str
    name: str
    subsystem: str = "air_intake_and_fuel_control"
    description: str


class FaultDefinition(BaseModel):
    fault_id: str
    name: str
    component_id: str
    description: str
    prior: float = Field(default=0.08, ge=0.0, le=1.0)
    # Expected DTCs with conditional likelihood P(DTC | Fault)
    dtcs: Dict[str, float] = Field(default_factory=dict)
    # Expected symptoms: symptom_name -> {state: P(state | Fault)}
    symptoms: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # Distinguishing physical or scanner tests
    distinguishing_tests: List[str] = Field(default_factory=list)


class Observations(BaseModel):
    """
    Observations provided by the technician or scanner.
    Missing observations are simply omitted; never treated as negative evidence.
    """
    dtc_codes: List[str] = Field(default_factory=list)
    symptoms: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None


class SimulationResult(BaseModel):
    fault_id: str
    fault_name: str
    component_id: str
    description: str
    expected_dtcs: Dict[str, float]
    expected_symptoms: Dict[str, Dict[str, float]]
    dominant_symptoms: Dict[str, str]
    distinguishing_tests: List[str]


class HypothesisScore(BaseModel):
    fault_id: str
    fault_name: str
    component_id: str
    prior: float
    log_likelihood: float
    posterior_prob: float
    evidence_matches: List[str] = Field(default_factory=list)
    evidence_contradictions: List[str] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    ranked_hypotheses: List[HypothesisScore]
    top_fault_id: Optional[str] = None
    confidence_gap: float = 0.0
    is_ambiguous: bool = False
    insufficient_evidence: bool = False
    suggested_tests: List[str] = Field(default_factory=list)
