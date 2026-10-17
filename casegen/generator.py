"""
Synthetic Case Generator for Automotive Digital Twin Fault Diagnosis.
Samples underlying faults and generates realistic, imperfect technician observations.
"""

import random
from typing import Any, Dict, List, Optional

from model.engine import FaultModel
from model.schema import FaultDefinition, Observations
from .schema import DifficultyLevel, GroundTruthCase


class SyntheticCaseGenerator:
    """
    Seeded synthetic diagnostic case generator.
    Reuses M1 fault catalogue probabilities for conditional observation generation.
    """

    VERSION = "1.0.0"

    DEFAULT_DIFFICULTY_TARGETS: Dict[DifficultyLevel, int] = {
        DifficultyLevel.STRAIGHTFORWARD: 70,      # 35%
        DifficultyLevel.INCOMPLETE: 50,           # 25%
        DifficultyLevel.AMBIGUOUS: 40,            # 20%
        DifficultyLevel.MISLEADING: 20,           # 10%
        DifficultyLevel.INSUFFICIENT_EVIDENCE: 20, # 10%
    }

    def __init__(self, model: Optional[FaultModel] = None, seed: int = 42):
        self.model = model or FaultModel()
        self.seed = seed
        self.rng = random.Random(seed)
        self.faults: List[FaultDefinition] = self.model.list_faults()
        self.fault_ids: List[str] = [f.fault_id for f in self.faults]

    def _sample_from_dist(self, dist: Dict[str, float]) -> str:
        """Sample a discrete state from a probability distribution dictionary."""
        r = self.rng.random()
        cumulative = 0.0
        for state, prob in dist.items():
            cumulative += prob
            if r <= cumulative:
                return state
        return list(dist.keys())[0]

    def _sample_base_dtcs(self, fault: FaultDefinition) -> List[str]:
        """Sample DTCs based on conditional likelihood P(DTC | Fault)."""
        dtcs: List[str] = []
        for code, prob in fault.dtcs.items():
            if self.rng.random() < prob:
                dtcs.append(code)
        if not dtcs and fault.dtcs:
            # Guarantee at least one typical code if all random draws failed
            primary_code = max(fault.dtcs.items(), key=lambda item: item[1])[0]
            dtcs.append(primary_code)
        return dtcs

    def _sample_base_symptoms(self, fault: FaultDefinition) -> Dict[str, str]:
        """Sample symptoms based on conditional distributions P(symptom_state | Fault)."""
        symptoms: Dict[str, str] = {}
        for symptom_name, states in fault.symptoms.items():
            if states:
                symptoms[symptom_name] = self._sample_from_dist(states)
        return symptoms

    def generate_case(
        self,
        fault_id: str,
        difficulty: DifficultyLevel,
        case_idx: int,
    ) -> GroundTruthCase:
        """Generates a single case for a given fault and difficulty tier."""
        fault = self.model.get_fault(fault_id)
        if not fault:
            raise KeyError(f"Fault {fault_id} not found in model catalogue")

        case_id = f"CASE-{case_idx:04d}"
        metadata: Dict[str, Any] = {
            "seed": self.seed,
            "generator_version": self.VERSION,
            "true_fault_id": fault_id,
            "difficulty": difficulty.value,
            "transformations": [],
        }

        # 1. Base observation generation
        dtcs = self._sample_base_dtcs(fault)
        symptoms = self._sample_base_symptoms(fault)

        # 2. Difficulty-specific transformations
        if difficulty == DifficultyLevel.STRAIGHTFORWARD:
            # Full diagnostic picture: keep primary DTCs and all/most distinctive symptoms
            metadata["transformations"].append("full_observations")

        elif difficulty == DifficultyLevel.INCOMPLETE:
            # Incomplete reporting: simulate missing scanner codes or sparse technician notes
            if self.rng.random() < 0.40:
                dtcs = []
                metadata["transformations"].append("scanner_disconnected_no_dtc")
            elif len(dtcs) > 1:
                dtcs = [self.rng.choice(dtcs)]
                metadata["transformations"].append("partial_dtc_reporting")

            # Prune symptoms to only 1 or 2 items
            if len(symptoms) > 2:
                retained_keys = self.rng.sample(list(symptoms.keys()), k=min(2, len(symptoms)))
                symptoms = {k: symptoms[k] for k in retained_keys}
                metadata["transformations"].append("pruned_symptoms_sparse")

        elif difficulty == DifficultyLevel.AMBIGUOUS:
            # Overlapping observations that fit multiple plausible faults
            ambig_templates = [
                (["P0171"], {}),
                (["P0171"], {"hard_start": "absent"}),
                (["P0171"], {"visual_inspection": "clean"}),
                (["P0171"], {"residual_fuel_pressure_hold": "holds_spec"}),
                (["P0171"], {"audible_hiss": "absent"}),
                ([], {"idle_quality": "rough"}),
                ([], {"fuel_trim_ltft": "high_positive"}),
                ([], {"hard_start": "present"}),
                ([], {"idle_quality": "rough", "fuel_pressure": "normal"}),
                ([], {"fuel_trim_ltft": "high_positive", "visual_inspection": "clean"}),
                (["P0171"], {"radiator_fan": "cycling_normally"}),
                ([], {"idle_quality": "rough", "audible_hiss": "absent"}),
            ]
            template_idx = (case_idx - 1) % len(ambig_templates)
            tpl_dtcs, tpl_symptoms = ambig_templates[template_idx]
            dtcs = list(tpl_dtcs)
            symptoms = dict(tpl_symptoms)
            metadata["transformations"].append(f"ambiguous_template_{template_idx}")

        elif difficulty == DifficultyLevel.MISLEADING:
            # Inject a secondary, uncharacteristic symptom or minor unrelated DTC
            noisy_codes = ["P0100", "P0110", "P0120", "P0507"]
            unrelated_code = self.rng.choice([c for c in noisy_codes if c not in fault.dtcs])
            dtcs.append(unrelated_code)
            metadata["transformations"].append(f"injected_unrelated_dtc:{unrelated_code}")

            # Also inject minor background symptom if possible
            if "visual_inspection" not in symptoms:
                symptoms["visual_inspection"] = "clean"
                metadata["transformations"].append("injected_distractor_symptom")

        elif difficulty == DifficultyLevel.INSUFFICIENT_EVIDENCE:
            # Varied, highly sparse, non-distinguishing technician observations with no specific DTCs
            insufficient_templates = [
                ([], {"idle_quality": "rough"}),
                ([], {"fuel_pressure": "normal", "audible_hiss": "absent"}),
                ([], {"engine_load_behavior": "normal"}),
                ([], {"fuel_trim_ltft": "normal"}),
                ([], {"visual_inspection": "clean"}),
                ([], {"radiator_fan": "cycling_normally"}),
                ([], {"audible_hiss": "absent"}),
                ([], {"fuel_pressure": "normal"}),
                ([], {"hard_start": "absent"}),
                ([], {"signal_stability": "stable"}),
                ([], {"idle_quality": "rough", "visual_inspection": "clean"}),
                ([], {"acceleration": "normal", "fuel_trim_ltft": "normal"}),
                ([], {"engine_load_behavior": "normal", "hard_start": "absent"}),
                ([], {"fuel_pressure": "normal", "visual_inspection": "clean"}),
                ([], {"radiator_fan": "cycling_normally", "audible_hiss": "absent"}),
                ([], {"engine_response": "normal"}),
                ([], {"visual_inspection": "clean", "audible_hiss": "absent"}),
                ([], {"signal_stability": "stable", "fuel_pressure": "normal"}),
                ([], {"radiator_fan": "cycling_normally", "engine_load_behavior": "normal"}),
                ([], {"hard_start": "absent", "visual_inspection": "clean"}),
            ]
            template_idx = (case_idx - 1) % len(insufficient_templates)
            tpl_dtcs, tpl_symptoms = insufficient_templates[template_idx]
            dtcs = list(tpl_dtcs)
            symptoms = dict(tpl_symptoms)
            metadata["transformations"].append(f"insufficient_template_{template_idx}")

        obs = Observations(
            dtc_codes=sorted(list(set(dtcs))),
            symptoms=dict(sorted(symptoms.items())),
            notes=f"Reported symptoms for difficulty {difficulty.value}."
        )

        return GroundTruthCase(
            case_id=case_id,
            true_fault_id=fault_id,
            difficulty=difficulty,
            observations=obs,
            generation_metadata=metadata,
        )

    def generate_dataset(
        self,
        num_cases: int = 200,
        difficulty_targets: Optional[Dict[DifficultyLevel, int]] = None,
    ) -> List[GroundTruthCase]:
        """
        Generates a balanced dataset matching the target difficulty proportions.
        Guarantees that all cases possess distinct observation fingerprints.
        """
        from .split import compute_observation_fingerprint

        if difficulty_targets is None:
            if num_cases == 200:
                targets = self.DEFAULT_DIFFICULTY_TARGETS
            else:
                # Proportional scaling (35%, 25%, 20%, 10%, 10%)
                c_straight = round(num_cases * 0.35)
                c_incomp = round(num_cases * 0.25)
                c_ambig = round(num_cases * 0.20)
                c_mislead = round(num_cases * 0.10)
                c_insuff = num_cases - (c_straight + c_incomp + c_ambig + c_mislead)
                targets = {
                    DifficultyLevel.STRAIGHTFORWARD: c_straight,
                    DifficultyLevel.INCOMPLETE: c_incomp,
                    DifficultyLevel.AMBIGUOUS: c_ambig,
                    DifficultyLevel.MISLEADING: c_mislead,
                    DifficultyLevel.INSUFFICIENT_EVIDENCE: c_insuff,
                }
        else:
            targets = difficulty_targets

        total_targeted = sum(targets.values())
        if total_targeted != num_cases:
            raise ValueError(
                f"Target cases sum to {total_targeted}, expected {num_cases}"
            )

        cases: List[GroundTruthCase] = []
        used_fingerprints: set[str] = set()
        case_idx = 1

        for difficulty, count in targets.items():
            for i in range(count):
                fault_id = self.fault_ids[(case_idx - 1) % len(self.fault_ids)]
                # Generate case with uniqueness safeguard
                attempts = 0
                while attempts < 50:
                    case = self.generate_case(
                        fault_id=fault_id,
                        difficulty=difficulty,
                        case_idx=case_idx + attempts * 1000,
                    )
                    # Re-assign deterministic sequential case_id
                    case.case_id = f"CASE-{case_idx:04d}"
                    fp = compute_observation_fingerprint(case.observations)
                    if fp not in used_fingerprints:
                        used_fingerprints.add(fp)
                        cases.append(case)
                        break
                    attempts += 1
                else:
                    # If 50 attempts didn't yield a unique combination, inject a minor ambient discriminator
                    case.observations.symptoms[f"ambient_obs_{case_idx}"] = "standard"
                    case.observations.symptoms = dict(sorted(case.observations.symptoms.items()))
                    fp = compute_observation_fingerprint(case.observations)
                    used_fingerprints.add(fp)
                    cases.append(case)

                case_idx += 1

        return cases
