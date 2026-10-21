"""
Fault simulation and likelihood-based scoring engine for automotive diagnosis.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .schema import (
    Component,
    DiagnosticResult,
    FaultDefinition,
    HypothesisScore,
    Observations,
    SimulationResult,
)
from .vehicle_registry import vehicle_registry


class FaultModel:
    """
    Diagnostic model for generic petrol air-intake and fuel-control subsystem.
    Implements transparent likelihood-based scoring and fault simulation.
    """

    def __init__(self, catalogue_path: Optional[str] = None):
        if catalogue_path is None:
            catalogue_path = str(Path(__file__).parent / "catalogue.yaml")

        self.catalogue_path = Path(catalogue_path)
        self.components: Dict[str, Component] = {}
        self.faults: Dict[str, FaultDefinition] = {}
        self._load_catalogue()

    def _load_catalogue(self) -> None:
        candidate_paths = [
            self.catalogue_path,
            Path(__file__).parent / "catalogue.yaml",
            Path.cwd() / "model" / "catalogue.yaml",
            Path("/home/user/app/model/catalogue.yaml"),
        ]
        target_path = None
        for p in candidate_paths:
            if p.exists():
                target_path = p
                break

        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"Catalogue file not found at: {self.catalogue_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for comp_data in data.get("components", []):
            comp = Component(**comp_data)
            self.components[comp.id] = comp

        for fault_data in data.get("faults", []):
            fault = FaultDefinition(**fault_data)
            self.faults[fault.fault_id] = fault

    def get_component(self, component_id: str) -> Optional[Component]:
        return self.components.get(component_id)

    def get_fault(self, fault_id: str) -> Optional[FaultDefinition]:
        return self.faults.get(fault_id)

    def list_components(self) -> List[Component]:
        return list(self.components.values())

    def list_faults(self) -> List[FaultDefinition]:
        return list(self.faults.values())

    def simulate_fault(self, fault_id: str, vehicle_id: str = "generic_petrol") -> SimulationResult:
        """
        Simulate a fault and return expected symptoms, DTCs, and dominant behavior.
        Supports vehicle profile overlays for platform-specific faults.
        """
        fault = self.get_fault(fault_id)
        if not fault:
            profile = vehicle_registry.get_profile(vehicle_id)
            if profile and profile.additional_faults:
                for extra in profile.additional_faults:
                    if extra.get("fault_id") == fault_id:
                        fault = FaultDefinition(**extra)
                        break
        if not fault:
            raise KeyError(f"Fault '{fault_id}' not found in catalogue")

        dominant: Dict[str, str] = {}
        for symptom_name, states in fault.symptoms.items():
            if states:
                best_state = max(states.items(), key=lambda item: item[1])[0]
                dominant[symptom_name] = best_state

        comp = self.get_component(fault.component_id)
        subsystem = comp.subsystem if comp else "air_intake_and_fuel_control"
        return SimulationResult(
            fault_id=fault.fault_id,
            fault_name=fault.name,
            component_id=fault.component_id,
            subsystem=subsystem,
            description=fault.description,
            expected_dtcs=fault.dtcs,
            expected_symptoms=fault.symptoms,
            dominant_symptoms=dominant,
            distinguishing_tests=fault.distinguishing_tests,
            cascading_effects=getattr(fault, "cascading_effects", []),
        )

    # Baseline frequencies of observations in healthy/ambient condition
    BASELINE_DTC = 0.02
    NORMAL_STATES = {
        "normal", "none", "clean", "absent", "stable", "smooth",
        "holds_spec", "cycling_normally", "steady_voltage",
        "outlet_significantly_hotter", "misfire_stays_at_cylinder",
    }

    def _get_baseline_prob(self, state: str) -> float:
        return 0.85 if state.lower() in self.NORMAL_STATES else 0.08

    def score_hypotheses(
        self,
        observations: Observations,
        epsilon_contradiction: float = 0.01,
        epsilon_unrelated_dtc: float = 0.005,
    ) -> DiagnosticResult:
        """
        Scores candidate faults against provided observations using Log-Likelihood Ratio (Bayes Factor).

        Weight of Evidence: W(Fault : obs) = log( P(obs | Fault) / P_baseline(obs) )
        - Unmodeled/missing observations have zero weight (log(1.0) = 0.0).
        - Matching distinctive symptoms yield positive weights.
        - Contradictory observations yield negative penalties.
        """
        raw_scores: Dict[str, float] = {}
        evidence_matches: Dict[str, List[str]] = {}
        evidence_contradictions: Dict[str, List[str]] = {}

        num_observations = len(observations.dtc_codes) + len(observations.symptoms)

        # Retrieve vehicle profile for platform-specific priors and additional faults
        vehicle_id = getattr(observations, "vehicle_id", "generic_petrol") or "generic_petrol"
        profile = vehicle_registry.get_profile(vehicle_id)

        candidate_faults = dict(self.faults)
        if profile and profile.additional_faults:
            for extra in profile.additional_faults:
                extra_def = FaultDefinition(**extra)
                candidate_faults[extra_def.fault_id] = extra_def

        for fault_id, fault in candidate_faults.items():
            matches: List[str] = []
            contradictions: List[str] = []

            # Prior log probability (check vehicle-specific override if present)
            prior = fault.prior
            if profile and fault_id in profile.prior_overrides:
                prior = profile.prior_overrides[fault_id]
            prior = max(prior, 1e-4)
            log_score = math.log(prior)

            # 1. Evaluate observed DTCs
            for dtc in observations.dtc_codes:
                if dtc in fault.dtcs:
                    prob = fault.dtcs[dtc]
                    weight = math.log(max(prob, 1e-4) / self.BASELINE_DTC)
                    log_score += weight
                    matches.append(f"DTC {dtc} (P={prob:.2f}, W=+{weight:.2f})")
                else:
                    prob = epsilon_unrelated_dtc
                    weight = math.log(prob / self.BASELINE_DTC)
                    log_score += weight
                    contradictions.append(f"DTC {dtc} uncharacteristic (penalty: {weight:.2f})")

            # 2. Evaluate observed symptoms
            for symptom_name, observed_state in observations.symptoms.items():
                baseline = self._get_baseline_prob(observed_state)
                if symptom_name in fault.symptoms:
                    state_dist = fault.symptoms[symptom_name]
                    if observed_state in state_dist:
                        prob = state_dist[observed_state]
                        weight = math.log(max(prob, epsilon_contradiction) / baseline)
                        log_score += weight
                        if weight > 0.1:
                            matches.append(f"{symptom_name}={observed_state} (W=+{weight:.2f})")
                        elif weight < -0.5:
                            contradictions.append(
                                f"{symptom_name}={observed_state} contradicts pattern (penalty: {weight:.2f})"
                            )
                    else:
                        weight = math.log(epsilon_contradiction / baseline)
                        log_score += weight
                        contradictions.append(
                            f"{symptom_name}={observed_state} unexpected state (penalty: {weight:.2f})"
                        )
                # Unmodeled symptoms have P(s=v | Fault) == baseline, so weight = log(1.0) = 0.0 (neutral)

            raw_scores[fault_id] = log_score
            evidence_matches[fault_id] = matches
            evidence_contradictions[fault_id] = contradictions

        # Softmax normalization over candidates to produce inspectable posterior probabilities
        max_log = max(raw_scores.values()) if raw_scores else 0.0
        exp_scores = {fid: math.exp(score - max_log) for fid, score in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        posteriors = {fid: exp_scores[fid] / sum_exp for fid in exp_scores}

        # Build sorted list of HypothesisScore
        ranked: List[HypothesisScore] = []
        for fault_id, score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True):
            fault = candidate_faults[fault_id]
            comp = self.get_component(fault.component_id)
            subsystem = comp.subsystem if comp else "air_intake_and_fuel_control"
            prior_val = fault.prior
            if profile and fault_id in profile.prior_overrides:
                prior_val = profile.prior_overrides[fault_id]
            cascading_risks = [
                f"Risk of secondary {e.get('target_fault')}: {e.get('mechanism')}"
                for e in getattr(fault, "cascading_effects", [])
            ]
            ranked.append(
                HypothesisScore(
                    fault_id=fault_id,
                    fault_name=fault.name,
                    component_id=fault.component_id,
                    subsystem=subsystem,
                    prior=prior_val,
                    log_likelihood=round(score, 4),
                    posterior_prob=round(posteriors[fault_id], 4),
                    evidence_matches=evidence_matches[fault_id],
                    evidence_contradictions=evidence_contradictions[fault_id],
                    cascading_risks=cascading_risks,
                )
            )

        # Ambiguity and uncertainty evaluation
        top1 = ranked[0] if ranked else None
        top2 = ranked[1] if len(ranked) > 1 else None
        confidence_gap = (top1.posterior_prob - top2.posterior_prob) if (top1 and top2) else 1.0

        insufficient_evidence = False
        is_ambiguous = False

        if num_observations == 0:
            insufficient_evidence = True
            is_ambiguous = True
        else:
            if confidence_gap < 0.15:
                is_ambiguous = True
            if top1 and top1.posterior_prob < 0.25:
                insufficient_evidence = True

        # Aggregate suggested next tests from the top 3 hypotheses
        suggested_tests: List[str] = []
        for hyp in ranked[:3]:
            f = candidate_faults[hyp.fault_id]
            for test in f.distinguishing_tests:
                if test not in suggested_tests:
                    suggested_tests.append(test)

        return DiagnosticResult(
            ranked_hypotheses=ranked,
            top_fault_id=top1.fault_id if top1 else None,
            confidence_gap=round(confidence_gap, 4),
            is_ambiguous=is_ambiguous,
            insufficient_evidence=insufficient_evidence,
            suggested_tests=suggested_tests,
        )
