"""
Automated Test Suite for Phase 4: EV & Hybrid Electrified Powertrain Subsystem.
Verifies High-Voltage Battery, BMS, Traction Inverter, and PMSM Motor diagnostics,
Toyota Prius HEV & Tesla Model 3 BEV platform configurations, EIG HV tests,
and Self-Review Gate compliance.
"""

import pytest

from agents.workflow import run_diagnosis
from model.engine import FaultModel
from model.schema import Observations
from model.vehicle_registry import vehicle_registry
from rag.store import KnowledgeBase


@pytest.fixture
def model():
    return FaultModel()


@pytest.fixture
def kb():
    return KnowledgeBase()


# -----------------------------------------------------------------------------
# 1. EV/Hybrid Subsystem & Component Topology
# -----------------------------------------------------------------------------

def test_ev_subsystem_catalogue_dimensions(model: FaultModel):
    components = model.list_components()
    subsystems = {c.subsystem for c in components}

    assert "high_voltage_battery_and_ev_powertrain" in subsystems

    ev_comps = [c.id for c in components if c.subsystem == "high_voltage_battery_and_ev_powertrain"]
    assert "battery_management_system" in ev_comps
    assert "high_voltage_battery_pack" in ev_comps
    assert "traction_inverter" in ev_comps
    assert "electric_traction_motor" in ev_comps


def test_ev_vehicle_profiles_registered():
    vehicle_ids = vehicle_registry.list_vehicle_ids()
    assert "toyota_prius_18_hybrid" in vehicle_ids
    assert "tesla_model_3_bev" in vehicle_ids

    prius = vehicle_registry.get_profile("toyota_prius_18_hybrid")
    assert prius.make == "Toyota"
    assert prius.fuel_system == "HEV"
    assert "hv_pack_voltage_nominal" in prius.nominal_specs
    assert prius.nominal_specs["hv_pack_voltage_nominal"].min == 201.6

    tesla = vehicle_registry.get_profile("tesla_model_3_bev")
    assert tesla.make == "Tesla"
    assert tesla.fuel_system == "BEV"
    assert "hv_isolation_min_mohm" in tesla.nominal_specs
    assert tesla.nominal_specs["hv_isolation_min_mohm"].min == 10.0


# -----------------------------------------------------------------------------
# 2. EV / Hybrid Fault Scoring & Platform Overlays
# -----------------------------------------------------------------------------

def test_prius_inverter_cooling_pump_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0A93"],
        symptoms={
            "battery_cooling_flow": "no_circulation",
            "inverter_temp": "abnormally_high",
        },
        vehicle_id="toyota_prius_18_hybrid",
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "battery_cooling_pump_failure"
    assert top_fault.subsystem == "high_voltage_battery_and_ev_powertrain"
    assert top_fault.posterior_prob > 0.85


def test_tesla_hv_isolation_loss_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0AA6"],
        symptoms={
            "hv_isolation_megohmmeter": "low_isolation_fault",
            "vehicle_ready_state": "no_ready_interlock_open",
        },
        vehicle_id="tesla_model_3_bev",
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id in ["high_voltage_isolation_loss", "tesla_pyro_fuse_blown"]
    assert top_fault.subsystem == "high_voltage_battery_and_ev_powertrain"
    assert top_fault.posterior_prob > 0.80


def test_bms_cell_voltage_imbalance_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0A7F"],
        symptoms={
            "cell_voltage_delta": "excessive_imbalance",
            "ev_acceleration": "derated",
        },
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "bms_cell_voltage_imbalance"
    assert top_fault.subsystem == "high_voltage_battery_and_ev_powertrain"
    assert top_fault.posterior_prob > 0.85


def test_traction_motor_resolver_fault_scoring(model: FaultModel):
    obs = Observations(
        dtc_codes=["P0A40"],
        symptoms={
            "resolver_excitation_waveform": "distorted_or_missing",
            "vehicle_ready_state": "drive_inhibited",
        },
    )
    result = model.score_hypotheses(obs)

    top_fault = result.ranked_hypotheses[0]
    assert top_fault.fault_id == "traction_motor_resolver_fault"
    assert top_fault.subsystem == "high_voltage_battery_and_ev_powertrain"
    assert top_fault.posterior_prob > 0.85


# -----------------------------------------------------------------------------
# 3. Targeted EV/Hybrid RAG Retrieval
# -----------------------------------------------------------------------------

def test_prius_and_tesla_rag_tsbs(kb: KnowledgeBase):
    # Prius inverter pump TSB
    hits_prius = kb.search("Prius P0A93 inverter electric water pump", vehicle_id="toyota_prius_18_hybrid", top_k=3)
    assert len(hits_prius) > 0
    assert hits_prius[0].doc_id == "DOC-TSB-PRIUS-2021"

    # Tesla penthouse isolation TSB
    hits_tesla = kb.search("Tesla Model 3 P0AA6 penthouse moisture isolation", vehicle_id="tesla_model_3_bev", top_k=3)
    assert len(hits_tesla) > 0
    assert hits_tesla[0].doc_id == "DOC-TSB-TESLA-2022"


# -----------------------------------------------------------------------------
# 4. End-to-End EV LangGraph Diagnosis & Self-Review Gate
# -----------------------------------------------------------------------------

def test_e2e_prius_hybrid_diagnosis():
    obs = Observations(
        dtc_codes=["P0A93"],
        symptoms={
            "battery_cooling_flow": "no_circulation",
            "inverter_temp": "abnormally_high",
        },
        vehicle_id="toyota_prius_18_hybrid",
    )
    report = run_diagnosis(obs)

    assert report.vehicle_id == "toyota_prius_18_hybrid"
    assert report.review_status == "PASSED"
    assert report.confidence_score > 0.85
    assert any("P0A93" in c for c in report.citations)


def test_e2e_tesla_bev_diagnosis():
    obs = Observations(
        dtc_codes=["P0AA6"],
        symptoms={
            "hv_isolation_megohmmeter": "low_isolation_fault",
            "vehicle_ready_state": "no_ready_interlock_open",
        },
        vehicle_id="tesla_model_3_bev",
    )
    report = run_diagnosis(obs)

    assert report.vehicle_id == "tesla_model_3_bev"
    assert report.review_status == "PASSED"
    assert report.confidence_score > 0.80
    assert any("P0AA6" in c for c in report.citations)
