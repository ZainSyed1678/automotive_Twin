"""
Unit and integration tests for Milestone 5: Adaptive Questioning & Self-Review Gate.
Verifies Expected Information Gain (EIG), stopping criteria, review gate validation,
and interactive multi-turn diagnostic disambiguation.
"""

import pytest
from model.engine import FaultModel
from model.schema import Observations
from agents.eig import (
    calculate_entropy,
    compute_eig_for_test,
    select_best_follow_up_question,
    TEST_QUESTION_METADATA,
)
from agents.review import review_diagnostic_report, ReviewResult
from agents.workflow import run_diagnosis, run_interactive_diagnosis


@pytest.fixture
def fault_model():
    return FaultModel()


# -----------------------------------------------------------------------------
# Test 1: Information Theory & Entropy Calculation
# -----------------------------------------------------------------------------

def test_entropy_calculation():
    """Verify Shannon entropy computation against analytical values."""
    # Deterministic outcome: H = 0 bits
    assert calculate_entropy([1.0, 0.0]) == 0.0
    assert calculate_entropy([0.0, 1.0, 0.0]) == 0.0

    # Uniform 2-state: H = - 2 * (0.5 * log2(0.5)) = 1.0 bit
    assert pytest.approx(calculate_entropy([0.5, 0.5]), 1e-4) == 1.0

    # Uniform 4-state: H = - 4 * (0.25 * log2(0.25)) = 2.0 bits
    assert pytest.approx(calculate_entropy([0.25, 0.25, 0.25, 0.25]), 1e-4) == 2.0


# -----------------------------------------------------------------------------
# Test 2: EIG Prioritization on Ambiguous Lean Case (P0171)
# -----------------------------------------------------------------------------

def test_eig_prioritizes_fuel_pressure_on_lean_ambiguity(fault_model):
    """
    When P0171 is present with rough idle, intake_vacuum_leak and fuel_pump_weak
    closely compete. EIG must select a highly discriminative test (fuel_pressure
    or high-RPM fuel trim) over an irrelevant test (coolant_temp_scan).
    """
    obs = Observations(
        dtc_codes=["P0171"],
    )
    report = run_diagnosis(obs)

    assert report.is_ambiguous is True
    assert report.follow_up_question is not None

    q = report.follow_up_question
    assert q["test_id"] in ("fuel_pressure", "fuel_trim_at_high_rpm", "audible_hiss")
    assert q["eig_score"] > 0.10  # Significant entropy reduction
    assert len(q["possible_answers"]) >= 2
    assert "fuel" in q["question_text"].lower() or "rpm" in q["question_text"].lower() or "hiss" in q["question_text"].lower()


# -----------------------------------------------------------------------------
# Test 3: Stopping Condition on Unambiguous Case
# -----------------------------------------------------------------------------

def test_stopping_condition_on_confident_case():
    """
    When observations clearly distinguish a single fault (e.g. MAF contamination
    with P0101 and underreporting airflow), confidence is high and no question is asked.
    """
    obs = Observations(
        dtc_codes=["P0101"],
        symptoms={
            "maf_reading": "underreporting",
            "acceleration": "hesitation",
            "fuel_trim_ltft": "high_positive",
        },
    )
    report = run_diagnosis(obs)

    assert report.top_fault_id == "maf_contamination"
    assert report.confidence_score >= 0.70
    assert report.confidence_gap >= 0.25
    assert report.is_ambiguous is False
    # No follow-up question needed
    assert report.follow_up_question is None


# -----------------------------------------------------------------------------
# Test 4: Review Gate Passes on Legitimate Report
# -----------------------------------------------------------------------------

def test_review_gate_passes_on_valid_case(fault_model):
    """Verify that a standard diagnostic run passes all verification checks."""
    obs = Observations(
        dtc_codes=["P0171"],
        symptoms={"idle_quality": "rough"},
    )
    report = run_diagnosis(obs)

    assert report.review_status == "PASSED"
    assert len(report.review_notes) >= 2
    # Verify review notes contain scope and grounding confirmation
    notes_text = " ".join(report.review_notes)
    assert "Scope Check" in notes_text
    assert "Grounding Check" in notes_text


# -----------------------------------------------------------------------------
# Test 5: Review Gate Catches Out-of-Scope Component
# -----------------------------------------------------------------------------

def test_review_gate_catches_out_of_scope_component(fault_model):
    """Verify that candidate components outside the supported subsystem are flagged."""
    mock_state = {
        "observations": {"dtc_codes": ["P0171"], "symptoms": {}},
        "diagnostic_result": {
            "ranked_hypotheses": [
                {
                    "fault_id": "transmission_slipping",
                    "fault_name": "Transmission Slip",
                    "component_id": "torque_converter",  # Out of scope!
                    "posterior_prob": 0.8,
                    "evidence_matches": [],
                }
            ],
            "is_ambiguous": False,
            "insufficient_evidence": False,
        },
        "citations": [],
    }

    result: ReviewResult = review_diagnostic_report(mock_state, fault_model)

    assert result.scope_boundary_passed is False
    assert result.passed is False
    assert any("torque_converter" in c for c in result.unsupported_claims)
    assert result.review_status == "FLAGGED"


# -----------------------------------------------------------------------------
# Test 6: Review Gate Catches Ungrounded Evidence Matches
# -----------------------------------------------------------------------------

def test_review_gate_catches_ungrounded_evidence(fault_model):
    """Verify that claiming a match to an unobserved DTC or symptom is caught."""
    mock_state = {
        "observations": {"dtc_codes": ["P0171"], "symptoms": {"idle_quality": "rough"}},
        "diagnostic_result": {
            "ranked_hypotheses": [
                {
                    "fault_id": "fuel_pump_weak",
                    "fault_name": "Weak Fuel Pump",
                    "component_id": "fuel_pump",
                    "posterior_prob": 0.65,
                    # Claimed match for DTC P0087 which was NOT in observations!
                    "evidence_matches": ["DTC: P0087", "Symptom: idle_quality=rough"],
                }
            ],
            "is_ambiguous": False,
            "insufficient_evidence": False,
        },
        "citations": [],
    }

    result: ReviewResult = review_diagnostic_report(mock_state, fault_model)

    assert result.evidence_grounding_passed is False
    assert result.passed is False
    assert any("P0087" in c for c in result.unsupported_claims)


# -----------------------------------------------------------------------------
# Test 7: Review Gate Enforces Humility on Insufficient Evidence
# -----------------------------------------------------------------------------

def test_review_gate_enforces_humility_on_empty_case(fault_model):
    """Verify that an empty observation set triggers insufficient evidence protection."""
    obs = Observations()
    report = run_diagnosis(obs)

    assert report.insufficient_evidence is True
    assert report.review_status == "PASSED"
    notes_text = " ".join(report.review_notes)
    assert "Humility Check" in notes_text


# -----------------------------------------------------------------------------
# Test 8: Interactive Multi-Turn Disambiguation Loop
# -----------------------------------------------------------------------------

def test_interactive_multi_turn_disambiguation():
    """
    Test the multi-turn conversational workflow:
    Turn 1: Ambiguous P0171 lean case. System generates follow-up test (e.g. fuel_pressure).
    Turn 2: Technician responds with 'low' fuel pressure.
    Outcome: Diagnostic certainty shifts decisively to 'fuel_pump_weak' and loop stops.
    """
    initial_obs = Observations(
        dtc_codes=["P0171"],
    )

    def technician_simulator(question_dict):
        # If asked about fuel pressure, report 'low'
        if question_dict["test_id"] == "fuel_pressure":
            return "low"
        # If asked about high rpm fuel trim, report 'worsens'
        if question_dict["test_id"] == "fuel_trim_at_high_rpm":
            return "worsens"
        # If asked about hiss, report 'absent'
        if question_dict["test_id"] == "audible_hiss":
            return "absent"
        return "normal"

    reports = run_interactive_diagnosis(
        initial_observations=initial_obs,
        answer_callback=technician_simulator,
        max_turns=3,
    )

    # Must take 2 turns to disambiguate
    assert len(reports) >= 2

    turn1 = reports[0]
    turn2 = reports[1]

    # Turn 1 should be ambiguous with follow-up question
    assert turn1.is_ambiguous is True
    assert turn1.follow_up_question is not None

    # Turn 2 should have resolved the ambiguity towards fuel_pump_weak
    assert turn2.top_fault_id == "fuel_pump_weak"
    assert turn2.confidence_score > turn1.confidence_score
    assert turn2.confidence_gap > turn1.confidence_gap


# -----------------------------------------------------------------------------
# Test 9: Interactive Multi-Turn Respects Max Turns
# -----------------------------------------------------------------------------

def test_interactive_multi_turn_respects_max_turns():
    """Verify that the interactive loop halts strictly at max_turns even if ambiguous."""
    initial_obs = Observations(dtc_codes=["P0171"])

    def uncooperative_technician(question_dict):
        return "normal"

    reports = run_interactive_diagnosis(
        initial_observations=initial_obs,
        answer_callback=uncooperative_technician,
        max_turns=2,
    )

    assert len(reports) <= 2
