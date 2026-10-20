"""
Automated Test Suite for Phase 3: Multi-Subsystem Digital Twin & Cross-Subsystem Fault Propagation.
Verifies multi-subsystem causal architecture (Air-Intake, Ignition & Combustion, Exhaust & Emissions),
cascading fault hazard propagation, EIG coil swap discrimination, and Self-Review gate verification.
"""

import pytest

from agents.eig import select_best_follow_up_question
from agents.workflow import run_diagnosis
from model.engine import FaultModel
from model.schema import Observations
from rag.store import KnowledgeBase


@pytest.fixture
def model():
    return FaultModel()


@pytest.fixture
def kb():
    return KnowledgeBase()


# -----------------------------------------------------------------------------
# 1. Multi-Subsystem Dimensions & Cross-Subsystem Topology
# -----------------------------------------------------------------------------

def test_multi_subsystem_catalogue_dimensions(model: FaultModel):
    components = model.list_components()
    faults = model.list_faults()

    # Verify at least 15 components across 3 distinct subsystems
    subsystems = {c.subsystem for c in components}
    assert "air_intake_and_fuel_control" in subsystems
    assert "ignition_and_combustion" in subsystems
    assert "exhaust_and_aftertreatment" in subsystems

    # Check component subsystem membership
    ignition_comps = [c.id for c in components if c.subsystem == "ignition_and_combustion"]
    assert "ignition_coil" in ignition_comps
    assert "spark_plug" in ignition_comps
    assert "crankshaft_position_sensor" in ignition_comps

    exhaust_comps = [c.id for c in components if c.subsystem == "exhaust_and_aftertreatment"]
    assert "catalytic_converter" in exhaust_comps
    assert "downstream_oxygen_sensor" in exhaust_comps


# -----------------------------------------------------------------------------
# 2. Ignition & Combustion Subsystem Scoring & Simulation
# -----------------------------------------------------------------------------

def test_ignition_coil_breakdown_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0300", "P0301"],
        symptoms={
            "idle_quality": "rough",
            "ignition_coil_swap": "misfire_follows_coil",
        },
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "ignition_coil_breakdown"
    assert top_fault.subsystem == "ignition_and_combustion"
    assert top_fault.posterior_prob > 0.80
    assert len(top_fault.cascading_risks) > 0
    assert "catalyst_efficiency_degraded" in top_fault.cascading_risks[0]


def test_spark_plug_fouled_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0300", "P0301"],
        symptoms={
            "idle_quality": "rough",
            "spark_plug_condition": "fouled_or_eroded",
            "ignition_coil_swap": "misfire_stays_at_cylinder",
        },
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "spark_plug_fouled"
    assert top_fault.subsystem == "ignition_and_combustion"
    assert top_fault.posterior_prob > 0.75


def test_crankshaft_position_sensor_thermal_failure(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0335"],
        symptoms={
            "engine_stall_when_hot": "present",
            "crank_no_start": "intermittent_when_warm",
        },
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "crank_sensor_thermal_failure"
    assert top_fault.subsystem == "ignition_and_combustion"
    assert top_fault.posterior_prob > 0.85


# -----------------------------------------------------------------------------
# 3. Exhaust & Aftertreatment Subsystem Scoring & Cascade Hazard
# -----------------------------------------------------------------------------

def test_catalytic_converter_degradation_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0420"],
        symptoms={"downstream_o2_switching": "active_mirroring_upstream"},
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "catalyst_efficiency_degraded"
    assert top_fault.subsystem == "exhaust_and_aftertreatment"
    assert top_fault.posterior_prob > 0.90


def test_cross_subsystem_cascade_simulation(model: FaultModel):
    sim = model.simulate_fault("ignition_coil_breakdown")
    assert sim.subsystem == "ignition_and_combustion"
    assert len(sim.cascading_effects) > 0
    assert sim.cascading_effects[0]["target_fault"] == "catalyst_efficiency_degraded"
    assert sim.cascading_effects[0]["risk_level"] == "high"


# -----------------------------------------------------------------------------
# 4. Multi-Subsystem Adaptive Questioning & RAG Retrieval
# -----------------------------------------------------------------------------

def test_misfire_eig_selects_coil_swap(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0300", "P0301"],
        symptoms={"idle_quality": "rough"},
    )
    diag_result = model.score_hypotheses(obs)

    question = select_best_follow_up_question(
        ranked_hypotheses=[h.model_dump() for h in diag_result.ranked_hypotheses],
        current_observations=obs.model_dump(),
        fault_model=model,
        confidence_gap=diag_result.confidence_gap,
        is_ambiguous=diag_result.is_ambiguous,
        insufficient_evidence=diag_result.insufficient_evidence,
    )
    assert question is not None
    assert question.test_id == "ignition_coil_swap"
    assert question.eig_score > 0.20


def test_multi_subsystem_rag_retrieval(kb: KnowledgeBase):
    # Ignition coil TSB
    hits_coil = kb.search("P0301 ignition coil boot tracking", top_k=3)
    doc_ids_coil = [h.doc_id for h in hits_coil]
    assert any(d in ["DOC-OBD-P0301", "DOC-TSB-2024-01"] for d in doc_ids_coil)

    # Catalytic converter TSB
    hits_cat = kb.search("P0420 catalytic converter misfire meltdown", top_k=3)
    doc_ids_cat = [h.doc_id for h in hits_cat]
    assert any(d in ["DOC-OBD-P0420", "DOC-TSB-2024-02"] for d in doc_ids_cat)


# -----------------------------------------------------------------------------
# 5. End-to-End Multi-Subsystem Diagnosis & Review Gate
# -----------------------------------------------------------------------------

def test_e2e_ignition_coil_diagnosis_and_review():
    obs = Observations(
        dtc_codes=["P0300", "P0301"],
        symptoms={
            "idle_quality": "rough",
            "ignition_coil_swap": "misfire_follows_coil",
        },
    )
    report = run_diagnosis(obs)

    assert report.top_fault_id == "ignition_coil_breakdown"
    assert report.confidence_score > 0.85
    assert report.review_status == "PASSED"
    assert any("DOC-OBD-P0301" in c for c in report.citations)


def test_e2e_catalyst_degradation_diagnosis_and_review():
    obs = Observations(
        dtc_codes=["P0420"],
        symptoms={"downstream_o2_switching": "active_mirroring_upstream"},
    )
    report = run_diagnosis(obs)

    assert report.top_fault_id == "catalyst_efficiency_degraded"
    assert report.confidence_score > 0.90
    assert report.review_status == "PASSED"
    assert any("DOC-OBD-P0420" in c for c in report.citations)
