"""
Unit and scenario tests for M1: Automotive Digital Twin Fault Model.
"""

import pytest
from model import FaultModel, Observations


@pytest.fixture
def model():
    return FaultModel()


def test_catalogue_structure_and_counts(model: FaultModel):
    """Verify that baseline and multi-subsystem components/faults are loaded with valid priors."""
    components = model.list_components()
    faults = model.list_faults()

    assert len(components) >= 10, f"Expected at least 10 components, got {len(components)}"
    assert len(faults) >= 12, f"Expected at least 12 faults, got {len(faults)}"

    # Verify baseline air-intake & fuel-control subsystem counts remain intact
    air_fuel_comps = [c for c in components if c.subsystem == "air_intake_and_fuel_control"]
    assert len(air_fuel_comps) == 10

    air_fuel_faults = [f for f in faults if model.get_component(f.component_id) and model.get_component(f.component_id).subsystem == "air_intake_and_fuel_control"]
    assert len(air_fuel_faults) == 12

    for fault in faults:
        assert 0.0 < fault.prior <= 1.0, f"Invalid prior for {fault.fault_id}"
        assert len(fault.distinguishing_tests) >= 1, f"Missing tests for {fault.fault_id}"
        assert model.get_component(fault.component_id) is not None, (
            f"Component {fault.component_id} not found"
        )


def test_simulate_fault(model: FaultModel):
    """Verify simulate_fault returns valid deterministic profiles and raises on unknown faults."""
    sim = model.simulate_fault("intake_vacuum_leak")
    assert sim.fault_id == "intake_vacuum_leak"
    assert "P0171" in sim.expected_dtcs
    assert sim.dominant_symptoms.get("fuel_trim_ltft") == "high_positive"
    assert sim.dominant_symptoms.get("idle_quality") == "rough"

    with pytest.raises(KeyError):
        model.simulate_fault("non_existent_fault")


def test_missing_evidence_neutrality(model: FaultModel):
    """
    Verify that unobserved symptoms are not treated as negative evidence.
    Technicians frequently provide only a DTC without sensor checks.
    """
    obs = Observations(dtc_codes=["P0171"])
    result = model.score_hypotheses(obs)

    # Vacuum leak and weak fuel pump both expect P0171; neither should be eliminated
    candidate_ids = [h.fault_id for h in result.ranked_hypotheses[:3]]
    assert "intake_vacuum_leak" in candidate_ids
    assert "fuel_pump_weak" in candidate_ids
    assert result.is_ambiguous is True, "Ambiguous DTC alone must flag ambiguity"


def test_contradictory_evidence_downranks(model: FaultModel):
    """
    Verify that contradictory evidence significantly penalizes incompatible hypotheses.
    """
    # 1. P0171 + fuel_pressure: normal -> weak fuel pump should be heavily penalized
    obs_normal_pressure = Observations(
        dtc_codes=["P0171"],
        symptoms={"fuel_pressure": "normal"}
    )
    result_normal = model.score_hypotheses(obs_normal_pressure)
    rank_order_normal = [h.fault_id for h in result_normal.ranked_hypotheses]

    assert rank_order_normal[0] == "intake_vacuum_leak"
    weak_pump_hyp = next(h for h in result_normal.ranked_hypotheses if h.fault_id == "fuel_pump_weak")
    assert len(weak_pump_hyp.evidence_contradictions) > 0
    assert "fuel_pump_weak" not in rank_order_normal[:2]

    # 2. P0171 + fuel_pressure: low -> weak fuel pump should rise to the top
    obs_low_pressure = Observations(
        dtc_codes=["P0171"],
        symptoms={"fuel_pressure": "low", "fuel_trim_at_high_rpm": "worsens"}
    )
    result_low = model.score_hypotheses(obs_low_pressure)
    assert result_low.ranked_hypotheses[0].fault_id == "fuel_pump_weak"


def test_insufficient_evidence_flag(model: FaultModel):
    """Verify that an empty observation triggers insufficient_evidence and ambiguity."""
    empty_obs = Observations()
    result = model.score_hypotheses(empty_obs)

    assert result.insufficient_evidence is True
    assert result.is_ambiguous is True


def test_handwritten_diagnostic_scenarios(model: FaultModel):
    """
    Test distinct realistic diagnostic cases to confirm domain accuracy.
    """
    # Scenario 1: Vacuum Leak with classic smoke/sound evidence
    res1 = model.score_hypotheses(
        Observations(
            dtc_codes=["P0171", "P0507"],
            symptoms={
                "fuel_trim_ltft": "high_positive",
                "fuel_trim_at_high_rpm": "normalizes",
                "audible_hiss": "present",
                "fuel_pressure": "normal",
            }
        )
    )
    assert res1.top_fault_id == "intake_vacuum_leak"
    assert res1.ranked_hypotheses[0].posterior_prob > 0.80
    assert res1.is_ambiguous is False

    # Scenario 2: ECT Sensor stuck cold
    res2 = model.score_hypotheses(
        Observations(
            dtc_codes=["P0117", "P0128"],
            symptoms={
                "live_coolant_temp": "sub_zero_or_frozen",
                "fuel_trim_ltft": "high_negative",
                "radiator_fan": "constantly_on",
            }
        )
    )
    assert res2.top_fault_id == "coolant_sensor_stuck_low"
    assert res2.ranked_hypotheses[0].posterior_prob > 0.90

    # Scenario 3: Throttle position sensor track wear
    res3 = model.score_hypotheses(
        Observations(
            dtc_codes=["P0121"],
            symptoms={
                "tps_voltage_sweep": "erratic_jumps",
                "idle_quality": "hunting",
            }
        )
    )
    assert res3.top_fault_id == "throttle_sensor_drift"
    assert res3.ranked_hypotheses[0].posterior_prob > 0.75

    # Scenario 4: MAP sensor calibration drift
    res4 = model.score_hypotheses(
        Observations(
            dtc_codes=["P0106"],
            symptoms={"map_reading_key_on_engine_off": "inaccurate"}
        )
    )
    assert res4.top_fault_id == "map_sensor_drift"
    assert res4.ranked_hypotheses[0].posterior_prob > 0.70
