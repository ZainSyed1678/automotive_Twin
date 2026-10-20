"""
Automated Test Suite for Phase 2: Multi-Vehicle Support.
Verifies vehicle profile schemas, declarative registries, vehicle-specific
causal scoring, GDI vs MPI physical tolerances, targeted RAG retrieval,
and end-to-end LangGraph diagnosis.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app
from agents.workflow import run_diagnosis
from model.engine import FaultModel
from model.schema import Observations
from model.vehicle_registry import vehicle_registry, VehicleRegistry
from rag.store import KnowledgeBase


@pytest.fixture
def fault_model():
    return FaultModel()


@pytest.fixture
def knowledge_base():
    return KnowledgeBase()


@pytest.fixture
def test_client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Vehicle Profile Registry & Schema Validation
# -----------------------------------------------------------------------------

def test_vehicle_registry_discovery():
    profiles = vehicle_registry.list_profiles()
    vehicle_ids = [p.id for p in profiles]

    assert "generic_petrol" in vehicle_ids
    assert "hyundai_i20_kappa_12" in vehicle_ids
    assert "vw_golf_14_tsi" in vehicle_ids

    # Validate default fallback
    fallback = vehicle_registry.get_profile("non_existent_vehicle")
    assert fallback is not None
    assert fallback.id == "generic_petrol"


def test_vehicle_profile_specifications():
    hyu = vehicle_registry.get_profile("hyundai_i20_kappa_12")
    assert hyu.make == "Hyundai"
    assert hyu.model == "i20"
    assert hyu.fuel_system == "MPI"
    assert hyu.displacement_l == 1.2
    assert "fuel_pressure_bar" in hyu.nominal_specs

    # Verify nominal tolerance helper
    spec = hyu.nominal_specs["fuel_pressure_bar"]
    assert spec.is_within_spec(3.4) is True
    assert spec.is_within_spec(2.1) is False

    vw = vehicle_registry.get_profile("vw_golf_14_tsi")
    assert vw.make == "Volkswagen"
    assert vw.model == "Golf"
    assert vw.fuel_system == "GDI"
    assert vw.aspiration == "Turbocharged"
    assert "fuel_pressure_high_bar" in vw.nominal_specs


# -----------------------------------------------------------------------------
# 2. Causal Engine Vehicle Overlays (Priors & Specific Faults)
# -----------------------------------------------------------------------------

def test_hyundai_evap_purge_valve_surfacing(fault_model):
    obs = Observations(
        dtc_codes=["P0171", "P0507"],
        symptoms={"idle_quality": "rough", "audible_hiss": "present"},
        vehicle_id="hyundai_i20_kappa_12",
    )
    result = fault_model.score_hypotheses(obs)

    fault_ids = [h.fault_id for h in result.ranked_hypotheses[:3]]
    assert "evap_purge_valve_stuck_open" in fault_ids

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id in ["evap_purge_valve_stuck_open", "intake_vacuum_leak"]


def test_vw_gdi_intake_valve_carbon_deposit(fault_model):
    obs = Observations(
        dtc_codes=["P0300"],
        symptoms={"idle_quality": "rough", "fuel_trim_ltft": "high_positive"},
        vehicle_id="vw_golf_14_tsi",
    )
    result = fault_model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "intake_valve_carbon_deposit"
    assert top_fault.posterior_prob > 0.50


def test_platform_specific_forward_simulation(fault_model):
    # Simulate Hyundai-specific fault
    sim_hyu = fault_model.simulate_fault("evap_purge_valve_stuck_open", vehicle_id="hyundai_i20_kappa_12")
    assert sim_hyu.fault_name == "EVAP Purge Control Solenoid Stuck Open"
    assert "P0171" in sim_hyu.expected_dtcs
    assert len(sim_hyu.distinguishing_tests) > 0

    # Simulate VW-specific fault
    sim_vw = fault_model.simulate_fault("intake_valve_carbon_deposit", vehicle_id="vw_golf_14_tsi")
    assert sim_vw.fault_name == "Intake Valve Carbon Fouling"
    assert "P0300" in sim_vw.expected_dtcs
    assert "borescope_intake_port_inspection" in sim_vw.distinguishing_tests


# -----------------------------------------------------------------------------
# 3. Vehicle-Targeted RAG Technical Bulletins
# -----------------------------------------------------------------------------

def test_vehicle_specific_rag_tsb_ranking(knowledge_base):
    # Hyundai TSB priority
    res_hyu = knowledge_base.search(
        query="P0171 lean rough idle purge",
        vehicle_id="hyundai_i20_kappa_12",
        top_k=3,
    )
    assert len(res_hyu) > 0
    assert res_hyu[0].doc_id == "DOC-TSB-HY-2018-04"
    assert "Hyundai" in res_hyu[0].title

    # VW TSI TSB priority
    res_vw = knowledge_base.search(
        query="P0300 cold misfire rough idle carbon",
        vehicle_id="vw_golf_14_tsi",
        top_k=3,
    )
    assert len(res_vw) > 0
    assert res_vw[0].doc_id == "DOC-TSB-VW-2019-12"
    assert "Volkswagen" in res_vw[0].title


# -----------------------------------------------------------------------------
# 4. End-to-End LangGraph Diagnostic Workflow with Vehicles
# -----------------------------------------------------------------------------

def test_e2e_hyundai_diagnosis():
    obs = Observations(
        dtc_codes=["P0171", "P0507"],
        symptoms={"idle_quality": "rough", "fuel_pressure": "normal"},
        vehicle_id="hyundai_i20_kappa_12",
    )
    report = run_diagnosis(obs)

    assert report.vehicle_id == "hyundai_i20_kappa_12"
    assert "Hyundai" in (report.vehicle_name or "")
    assert report.review_status == "PASSED"
    assert len(report.citations) >= 1
    assert any("DOC-TSB-HY-2018-04" in c for c in report.citations)


def test_e2e_vw_golf_diagnosis():
    obs = Observations(
        dtc_codes=["P0300"],
        symptoms={"idle_quality": "rough", "fuel_pressure": "normal"},
        vehicle_id="vw_golf_14_tsi",
    )
    report = run_diagnosis(obs)

    assert report.vehicle_id == "vw_golf_14_tsi"
    assert "Volkswagen" in (report.vehicle_name or "")
    assert report.top_fault_id == "intake_valve_carbon_deposit"
    assert report.review_status == "PASSED"
    assert any("DOC-TSB-VW-2019-12" in c for c in report.citations)


# -----------------------------------------------------------------------------
# 5. REST API Multi-Vehicle Endpoints
# -----------------------------------------------------------------------------

def test_api_vehicles_endpoints(test_client):
    # GET /vehicles
    resp = test_client.get("/vehicles")
    assert resp.status_code == 200
    vehicles = resp.json()
    assert len(vehicles) >= 3

    # GET /vehicles/{vehicle_id}
    resp_hyu = test_client.get("/vehicles/hyundai_i20_kappa_12")
    assert resp_hyu.status_code == 200
    data = resp_hyu.json()
    assert data["make"] == "Hyundai"
    assert data["fuel_system"] == "MPI"

    # POST /diagnose with vehicle_id
    payload = {
        "vehicle_id": "hyundai_i20_kappa_12",
        "dtc_codes": ["P0171", "P0507"],
        "symptoms": {"idle_quality": "rough"},
    }
    resp_diag = test_client.post("/diagnose", json=payload)
    assert resp_diag.status_code == 200
    diag_data = resp_diag.json()
    assert diag_data["vehicle_id"] == "hyundai_i20_kappa_12"
    assert diag_data["review_status"] == "PASSED"
