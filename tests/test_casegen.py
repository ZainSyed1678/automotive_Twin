"""
Automated test suite for M2: Diagnostic Case Generator.
Validates schemas, ground-truth isolation, reproducibility, split integrity, and diagnostic feasibility.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from casegen import (
    DifficultyLevel,
    GroundTruthCase,
    AgentCase,
    SyntheticCaseGenerator,
    compute_observation_fingerprint,
    split_dataset,
)
from model import FaultModel, Observations


# -----------------------------------------------------------------------------
# 1. Schema Validation Tests
# -----------------------------------------------------------------------------

def test_ground_truth_case_schema():
    """Verify valid ground-truth cases are accepted and missing required fields rejected."""
    obs = Observations(dtc_codes=["P0171"], symptoms={"fuel_pressure": "normal"})
    gt = GroundTruthCase(
        case_id="CASE-0001",
        true_fault_id="intake_vacuum_leak",
        difficulty=DifficultyLevel.STRAIGHTFORWARD,
        observations=obs,
    )
    assert gt.case_id == "CASE-0001"
    assert gt.true_fault_id == "intake_vacuum_leak"
    assert gt.difficulty == DifficultyLevel.STRAIGHTFORWARD

    # Reject missing case_id or true_fault_id
    with pytest.raises(ValidationError):
        GroundTruthCase(
            true_fault_id="intake_vacuum_leak",
            difficulty=DifficultyLevel.STRAIGHTFORWARD,
            observations=obs,
        )


def test_agent_case_forbids_extra_fields():
    """Verify AgentCase strictly forbids ground-truth or un-allowlisted attributes."""
    obs = Observations(dtc_codes=["P0171"], symptoms={"fuel_pressure": "normal"})
    agent_case = AgentCase(case_id="CASE-0001", observations=obs)
    assert agent_case.case_id == "CASE-0001"

    # Attempting to inject true_fault_id or difficulty must raise ValidationError
    with pytest.raises(ValidationError):
        AgentCase(
            case_id="CASE-0001",
            observations=obs,
            true_fault_id="intake_vacuum_leak",  # Forbidden!
        )

    with pytest.raises(ValidationError):
        AgentCase(
            case_id="CASE-0001",
            observations=obs,
            difficulty="straightforward",  # Forbidden!
        )


# -----------------------------------------------------------------------------
# 2. Ground-Truth Isolation Tests
# -----------------------------------------------------------------------------

def test_agent_projection_and_allowlist_serialization():
    """Verify that projecting or serializing a case never leaks ground-truth fields."""
    obs = Observations(dtc_codes=["P0171"], symptoms={"fuel_pressure": "normal"})
    gt = GroundTruthCase(
        case_id="CASE-0001",
        true_fault_id="intake_vacuum_leak",
        difficulty=DifficultyLevel.STRAIGHTFORWARD,
        observations=obs,
        generation_metadata={"private_note": "top_secret_eval_info"}
    )

    # 1. In-memory projection
    agent_case = gt.to_agent_case()
    assert not hasattr(agent_case, "true_fault_id")
    assert not hasattr(agent_case, "difficulty")
    assert not hasattr(agent_case, "generation_metadata")

    # 2. Serialized dictionary allowlist check
    agent_dict = gt.to_agent_dict()
    assert set(agent_dict.keys()) == {"case_id", "observations"}
    assert "true_fault_id" not in agent_dict
    assert "difficulty" not in agent_dict
    assert "generation_metadata" not in agent_dict


def test_exported_agent_files_contain_no_ground_truth():
    """Verify exported agent dataset files on disk contain zero ground truth keys."""
    agent_dev_path = Path("data/agent_cases/cases_dev.json")
    agent_eval_path = Path("data/agent_cases/cases_eval.json")

    for path in [agent_dev_path, agent_eval_path]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        assert len(cases) > 0
        forbidden_keys = {"true_fault_id", "difficulty", "generation_metadata", "prior"}
        for c in cases:
            found_forbidden = forbidden_keys.intersection(c.keys())
            assert not found_forbidden, f"Security leakage! Found forbidden keys: {found_forbidden} in {path}"
            assert set(c.keys()) == {"case_id", "observations"}


# -----------------------------------------------------------------------------
# 3. Reproducibility Tests
# -----------------------------------------------------------------------------

def test_generator_reproducibility():
    """Verify that identical seeds produce bitwise identical datasets."""
    gen1 = SyntheticCaseGenerator(seed=42)
    cases1 = gen1.generate_dataset(num_cases=40)

    gen2 = SyntheticCaseGenerator(seed=42)
    cases2 = gen2.generate_dataset(num_cases=40)

    assert len(cases1) == len(cases2)
    for c1, c2 in zip(cases1, cases2):
        assert c1.case_id == c2.case_id
        assert c1.true_fault_id == c2.true_fault_id
        assert c1.difficulty == c2.difficulty
        assert c1.observations.model_dump() == c2.observations.model_dump()


def test_different_seeds_produce_different_datasets():
    """Verify that distinct seeds produce distinct observations."""
    gen1 = SyntheticCaseGenerator(seed=42)
    cases1 = gen1.generate_dataset(num_cases=20)

    gen2 = SyntheticCaseGenerator(seed=999)
    cases2 = gen2.generate_dataset(num_cases=20)

    # Compare observation dumps
    dump1 = [c.observations.model_dump() for c in cases1]
    dump2 = [c.observations.model_dump() for c in cases2]
    assert dump1 != dump2


# -----------------------------------------------------------------------------
# 4. Fingerprint & Split Integrity Tests
# -----------------------------------------------------------------------------

def test_fingerprint_order_insensitivity():
    """Verify fingerprint is invariant to DTC or symptom key ordering."""
    obs1 = Observations(
        dtc_codes=["P0171", "P0507"],
        symptoms={"idle_quality": "rough", "fuel_pressure": "normal"}
    )
    # Reverse DTCs and symptom dictionary insertion order
    obs2 = Observations(
        dtc_codes=["P0507", "P0171"],
        symptoms={"fuel_pressure": "normal", "idle_quality": "rough"}
    )

    fp1 = compute_observation_fingerprint(obs1)
    fp2 = compute_observation_fingerprint(obs2)
    assert fp1 == fp2, "Fingerprint must be invariant to observation ordering"


def test_split_target_and_anti_leakage():
    """Verify 140/60 split targets and zero fingerprint overlap."""
    gen = SyntheticCaseGenerator(seed=42)
    cases = gen.generate_dataset(num_cases=200)

    dev_cases, eval_cases = split_dataset(cases, dev_target=140, eval_target=60, seed=42)

    assert len(dev_cases) == 140
    assert len(eval_cases) == 60

    dev_fps = {compute_observation_fingerprint(c.observations) for c in dev_cases}
    eval_fps = {compute_observation_fingerprint(c.observations) for c in eval_cases}
    intersection = dev_fps.intersection(eval_fps)

    assert len(intersection) == 0, f"Fingerprint leak across dev and eval: {len(intersection)} duplicates"


# -----------------------------------------------------------------------------
# 5. Diagnostic Feasibility Tests
# -----------------------------------------------------------------------------

def test_straightforward_cases_feasibility():
    """Verify that straightforward cases achieve high diagnostic accuracy under M1 model."""
    model = FaultModel()
    gen = SyntheticCaseGenerator(seed=42)
    cases = gen.generate_dataset(num_cases=50, difficulty_targets={DifficultyLevel.STRAIGHTFORWARD: 50})

    correct = 0
    for case in cases:
        result = model.score_hypotheses(case.observations)
        if result.top_fault_id == case.true_fault_id:
            correct += 1

    accuracy = correct / len(cases)
    assert accuracy >= 0.85, f"Expected >= 85% accuracy on straightforward cases, got {accuracy:.2%}"


def test_ambiguous_cases_feasibility():
    """Verify that ambiguous cases produce small confidence gaps and flag ambiguity."""
    model = FaultModel()
    gen = SyntheticCaseGenerator(seed=42)
    cases = gen.generate_dataset(num_cases=20, difficulty_targets={DifficultyLevel.AMBIGUOUS: 20})

    ambiguous_count = 0
    for case in cases:
        result = model.score_hypotheses(case.observations)
        if result.is_ambiguous or result.confidence_gap <= 0.15:
            ambiguous_count += 1

    assert ambiguous_count >= 15, (
        f"Expected at least 75% of ambiguous cases to flag ambiguity/low gap, got {ambiguous_count}/20"
    )


def test_insufficient_evidence_cases_feasibility():
    """Verify that insufficient-evidence cases do not falsely reveal confident diagnosis."""
    model = FaultModel()
    gen = SyntheticCaseGenerator(seed=42)
    cases = gen.generate_dataset(num_cases=20, difficulty_targets={DifficultyLevel.INSUFFICIENT_EVIDENCE: 20})

    flagged_or_low_confidence = 0
    for case in cases:
        result = model.score_hypotheses(case.observations)
        top_prob = result.ranked_hypotheses[0].posterior_prob if result.ranked_hypotheses else 0.0
        # Should be flagged as insufficient evidence, ambiguous, or have low probability
        if result.insufficient_evidence or result.is_ambiguous or top_prob < 0.35:
            flagged_or_low_confidence += 1

    assert flagged_or_low_confidence == 20, (
        f"All 20 insufficient-evidence cases must be flagged or have low top confidence, got {flagged_or_low_confidence}"
    )
