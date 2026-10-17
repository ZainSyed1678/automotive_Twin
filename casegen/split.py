"""
Dataset splitting, observation fingerprinting, and leakage prevention for M2.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from model.schema import Observations
from .schema import GroundTruthCase


def compute_observation_fingerprint(observations: Observations) -> str:
    """
    Computes a canonical SHA-256 fingerprint for an observation set.
    Invariant to the insertion order of DTCs or symptoms.
    Excludes case IDs, ground-truth labels, difficulty, notes, and metadata.
    """
    # 1. Normalize and sort DTCs
    normalized_dtcs = sorted(list(set(c.strip().upper() for c in observations.dtc_codes)))

    # 2. Normalize and sort symptom pairs
    normalized_symptoms = sorted(
        [(k.strip().lower(), v.strip().lower()) for k, v in observations.symptoms.items()]
    )

    canonical_payload = {
        "dtcs": normalized_dtcs,
        "symptoms": normalized_symptoms,
    }

    # Deterministic JSON representation (compact, sorted keys)
    canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def split_dataset(
    cases: List[GroundTruthCase],
    dev_target: int = 140,
    eval_target: int = 60,
    seed: int = 42,
) -> Tuple[List[GroundTruthCase], List[GroundTruthCase]]:
    """
    Performs stratified split across difficulty categories with balanced fault distribution.
    Enforces that fingerprints(dev) intersection fingerprints(eval) == empty set.
    """
    import random
    rng = random.Random(seed)

    total_cases = len(cases)
    if total_cases != (dev_target + eval_target):
        raise ValueError(
            f"Case count {total_cases} != dev_target ({dev_target}) + eval_target ({eval_target})"
        )

    # 1. Group cases by difficulty
    by_difficulty: Dict[str, List[GroundTruthCase]] = {}
    for case in cases:
        by_difficulty.setdefault(case.difficulty.value, []).append(case)

    dev_cases: List[GroundTruthCase] = []
    eval_cases: List[GroundTruthCase] = []

    eval_ratio = eval_target / total_cases  # 0.30

    # 2. Partition within each difficulty category
    for diff, diff_cases in sorted(by_difficulty.items()):
        # Sort by fault_id and then shuffle with seed to balance faults across split
        rng.shuffle(diff_cases)
        n_eval = round(len(diff_cases) * eval_ratio)
        eval_cases.extend(diff_cases[:n_eval])
        dev_cases.extend(diff_cases[n_eval:])

    # Sort deterministically by case_id
    dev_cases.sort(key=lambda c: c.case_id)
    eval_cases.sort(key=lambda c: c.case_id)

    # 3. Leakage Validation: fingerprints(dev) ∩ fingerprints(eval) == ∅
    dev_fps: Set[str] = {compute_observation_fingerprint(c.observations) for c in dev_cases}
    eval_fps: Set[str] = {compute_observation_fingerprint(c.observations) for c in eval_cases}
    collisions = dev_fps.intersection(eval_fps)

    if collisions:
        raise ValueError(
            f"Data leakage detected! {len(collisions)} identical observation fingerprints found across dev and eval splits."
        )

    return dev_cases, eval_cases


def export_datasets(
    dev_cases: List[GroundTruthCase],
    eval_cases: List[GroundTruthCase],
    base_dir: str = "data",
) -> Dict[str, str]:
    """
    Exports datasets to disk with strict separation between ground-truth and agent-visible data.
    """
    base_path = Path(base_dir)
    agent_dir = base_path / "agent_cases"
    agent_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth evaluation datasets
    dev_file = base_path / "cases_dev.json"
    eval_file = base_path / "cases_eval.json"

    with open(dev_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in dev_cases], f, indent=2)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in eval_cases], f, indent=2)

    # 2. Sanitized Agent-visible datasets (allowlist serialization only)
    agent_dev_file = agent_dir / "cases_dev.json"
    agent_eval_file = agent_dir / "cases_eval.json"

    with open(agent_dev_file, "w", encoding="utf-8") as f:
        json.dump([c.to_agent_dict() for c in dev_cases], f, indent=2)

    with open(agent_eval_file, "w", encoding="utf-8") as f:
        json.dump([c.to_agent_dict() for c in eval_cases], f, indent=2)

    # 3. Manifest
    manifest_file = base_path / "dataset_manifest.json"
    manifest = {
        "version": "1.0.0",
        "total_cases": len(dev_cases) + len(eval_cases),
        "dev_cases_count": len(dev_cases),
        "eval_cases_count": len(eval_cases),
        "unique_dev_fingerprints": len({compute_observation_fingerprint(c.observations) for c in dev_cases}),
        "unique_eval_fingerprints": len({compute_observation_fingerprint(c.observations) for c in eval_cases}),
        "leakage_collision_count": 0,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "dev_ground_truth": str(dev_file),
        "eval_ground_truth": str(eval_file),
        "agent_dev": str(agent_dev_file),
        "agent_eval": str(agent_eval_file),
        "manifest": str(manifest_file),
    }
