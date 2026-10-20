"""
Expected Information Gain (EIG) Question Selection Engine.
Selects targeted physical tests and technician questions to maximize entropy reduction
over competing fault hypotheses.
"""

import math
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from model.engine import FaultModel
from model.schema import FaultDefinition, Observations


class QuestionCandidate(BaseModel):
    """
    Structured representation of a follow-up diagnostic test question.
    """
    test_id: str
    test_name: str
    question_text: str
    possible_answers: List[str]
    eig_score: float = Field(..., description="Expected Information Gain in bits")
    rationale: str
    discriminates_between: List[str] = Field(default_factory=list)


# Question catalogue mapping observable symptom variables to technician-actionable questions
TEST_QUESTION_METADATA: Dict[str, Dict[str, Any]] = {
    "fuel_pressure": {
        "test_name": "Fuel Rail Pressure Test",
        "question_text": "Connect a mechanical gauge to the fuel rail test port. Is measured pressure normal, low, or high?",
        "possible_answers": ["normal", "low", "high"],
        "default_baseline": {"normal": 0.85, "low": 0.10, "high": 0.05},
        "rationale": "Directly isolates fuel delivery starvation (fuel pump/filter/regulator) from intake air leaks.",
    },
    "fuel_trim_at_high_rpm": {
        "test_name": "High-RPM Fuel Trim Test",
        "question_text": "Hold engine RPM at 2500 steady state. Do short-term and long-term fuel trims normalize, worsen, or remain unchanged?",
        "possible_answers": ["normalizes", "worsens", "unchanged"],
        "default_baseline": {"normalizes": 0.30, "worsens": 0.30, "unchanged": 0.40},
        "rationale": "Intake vacuum leaks normalize at high RPM due to increased air volume; fuel delivery starvation worsens.",
    },
    "audible_hiss": {
        "test_name": "Intake Vacuum Hiss Inspection",
        "question_text": "With engine idling and bonnet open, listen closely around the intake plenum and vacuum lines. Is an audible hissing sound present or absent?",
        "possible_answers": ["present", "absent"],
        "default_baseline": {"absent": 0.90, "present": 0.10},
        "rationale": "High-velocity unmetered air entering through cracked hoses produces a distinct audible hiss.",
    },
    "misfire_pattern": {
        "test_name": "Cylinder Misfire Pattern Scan",
        "question_text": "Examine live OBD misfire counters across all cylinders. Is the misfire cylinder-specific, random across cylinders, or none?",
        "possible_answers": ["cylinder_specific", "random", "none"],
        "default_baseline": {"none": 0.85, "cylinder_specific": 0.10, "random": 0.05},
        "rationale": "Distinguishes single-cylinder restrictions (clogged injector) from whole-bank air/fuel faults.",
    },
    "maf_reading": {
        "test_name": "MAF Live Data Inspection",
        "question_text": "Check live MAF sensor grams/second reading at warm idle and WOT. Is MAF reading underreporting, normal, or overreporting?",
        "possible_answers": ["normal", "underreporting", "overreporting"],
        "default_baseline": {"normal": 0.85, "underreporting": 0.10, "overreporting": 0.05},
        "rationale": "Hot-wire contamination causes chronic air mass under-reporting across the load range.",
    },
    "map_reading_key_on_engine_off": {
        "test_name": "KOEO MAP Sensor Barometric Check",
        "question_text": "Turn Key On with Engine Off (KOEO). Does the live MAP sensor reading match local ambient barometric pressure, or is it inaccurate?",
        "possible_answers": ["normal", "inaccurate"],
        "default_baseline": {"normal": 0.90, "inaccurate": 0.10},
        "rationale": "Uncalibrated or fatigued MAP sensor diaphragms register incorrect atmospheric pressure before crank.",
    },
    "o2_sensor_switching_rate": {
        "test_name": "Upstream O2 Sensor Switching Oscilloscope Test",
        "question_text": "Observe upstream O2 sensor (O2S1) waveform at 2000 RPM. Is the switching rate active and rapid, or sluggish/stuck?",
        "possible_answers": ["normal", "sluggish"],
        "default_baseline": {"normal": 0.85, "sluggish": 0.15},
        "rationale": "Silica or carbon poisoning severely slows zirconia switching time between 0.1V and 0.9V.",
    },
    "throttle_sweep_test": {
        "test_name": "Throttle Potentiometer Smooth Sweep Test",
        "question_text": "Sweep throttle pedal slowly with KOEO while observing TPS voltage on an oscilloscope. Does voltage rise smoothly or show dropouts/flat spots?",
        "possible_answers": ["normal", "dropouts_at_low_throttle"],
        "default_baseline": {"normal": 0.90, "dropouts_at_low_throttle": 0.10},
        "rationale": "Wiper track wear produces localized voltage dropouts at frequently used low-throttle positions.",
    },
    "connector_pin_inspection": {
        "test_name": "Wiring Harness Connector Visual & Resistance Inspection",
        "question_text": "Disconnect sensor plug and inspect terminal pins under lighting. Are pins clean and tight, or showing green oxidation/corrosion?",
        "possible_answers": ["clean", "corroded_green"],
        "default_baseline": {"clean": 0.90, "corroded_green": 0.10},
        "rationale": "Pin fretting and moisture ingress cause intermittent contact resistance and signal dropouts.",
    },
    "coolant_temp_scan": {
        "test_name": "Cold-Start ECT Temperature Sanity Check",
        "question_text": "Before starting cold engine, compare ECT scanner reading with ambient outdoor temperature. Does ECT match ambient or read abnormally low?",
        "possible_answers": ["normal", "abnormally_low"],
        "default_baseline": {"normal": 0.90, "abnormally_low": 0.10},
        "rationale": "An open circuit or broken thermistor element forces ECT reading to extreme sub-zero values.",
    },
    "fuel_pressure_bleed_down": {
        "test_name": "Fuel Rail Pressure Retention / Bleed-Down Test",
        "question_text": "Cycle ignition to prime fuel pump, then switch off. Does fuel pressure hold steady for 10 minutes or bleed down rapidly?",
        "possible_answers": ["holds_steady", "rapid_bleed_down"],
        "default_baseline": {"holds_steady": 0.85, "rapid_bleed_down": 0.15},
        "rationale": "A stuck-open pressure regulator or leaking check valve causes immediate rail pressure collapse.",
    },
    "ignition_coil_swap": {
        "test_name": "Ignition Coil Pack Cylinder Swap Test",
        "question_text": "Swap suspect cylinder coil pack to adjacent cylinder and clear codes. Does misfire follow coil pack, or stay at original cylinder?",
        "possible_answers": ["misfire_follows_coil", "misfire_stays_at_cylinder"],
        "default_baseline": {"misfire_stays_at_cylinder": 0.50, "misfire_follows_coil": 0.50},
        "rationale": "Definitively isolates an ignition coil internal insulation breakdown from spark plug or cylinder mechanical issues.",
    },
    "spark_plug_condition": {
        "test_name": "Spark Plug Visual & Gap Inspection",
        "question_text": "Remove spark plug and inspect electrode gap and ceramic nose. Is the plug clean, or fouled/eroded?",
        "possible_answers": ["clean", "fouled_or_eroded"],
        "default_baseline": {"clean": 0.85, "fouled_or_eroded": 0.15},
        "rationale": "Identifies carbon soot accumulation, fuel wash, or excessive ground gap causing spark quenching.",
    },
    "downstream_o2_switching": {
        "test_name": "Downstream O2S2 Voltage Switching Frequency Test",
        "question_text": "Graph downstream oxygen sensor (O2S2) voltage at 2500 RPM. Does voltage remain steady (0.55-0.75V) or actively switch mirroring upstream O2S1?",
        "possible_answers": ["steady_voltage", "active_mirroring_upstream", "fixed_bias"],
        "default_baseline": {"steady_voltage": 0.85, "active_mirroring_upstream": 0.10, "fixed_bias": 0.05},
        "rationale": "A functional catalytic converter stores oxygen and flatlines downstream O2; oscillating voltage confirms catalyst degradation.",
    },
    "catalytic_temp_delta": {
        "test_name": "Catalytic Converter Inlet vs Outlet Temperature Differential",
        "question_text": "Measure catalytic converter inlet and outlet temperatures with an infrared pyrometer. Is outlet significantly hotter (>25°C), or inlet hotter/same?",
        "possible_answers": ["outlet_significantly_hotter", "inlet_hotter_or_same"],
        "default_baseline": {"outlet_significantly_hotter": 0.85, "inlet_hotter_or_same": 0.15},
        "rationale": "Exothermic redox reactions inside a working converter cause outlet temperature to exceed inlet temperature.",
    },
}


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Computes Shannon entropy in bits for a discrete probability distribution.
    H(P) = -sum(p * log2(p)) for p > 0.
    """
    entropy = 0.0
    for p in probabilities:
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return max(0.0, entropy)


def compute_eig_for_test(
    test_id: str,
    faults: List[FaultDefinition],
    posterior_probs: Dict[str, float],
) -> float:
    """
    Computes Expected Information Gain (mutual information) for a candidate test variable.

    EIG(X) = H(P) - E_v [ H(P | X=v) ]
    """
    meta = TEST_QUESTION_METADATA.get(test_id)
    if not meta:
        return 0.0

    possible_answers = meta["possible_answers"]
    baseline_dist = meta["default_baseline"]

    # 1. Base entropy of current hypothesis distribution
    current_probs = [posterior_probs.get(f.fault_id, 0.0) for f in faults]
    total_prob = sum(current_probs)
    if total_prob <= 0:
        return 0.0

    normalized_current = [p / total_prob for p in current_probs]
    base_entropy = calculate_entropy(normalized_current)
    if base_entropy < 1e-4:
        return 0.0

    # 2. Compute marginal probability P(X = v) and conditional posterior P(F_i | X = v)
    # P(v) = sum_i P(F_i) * P(v | F_i)
    p_outcomes: Dict[str, float] = {v: 0.0 for v in possible_answers}
    p_fault_given_outcome: Dict[str, Dict[str, float]] = {v: {} for v in possible_answers}

    for fault, p_fault in zip(faults, normalized_current):
        symptom_dist = fault.symptoms.get(test_id, baseline_dist)
        for v in possible_answers:
            likelihood = symptom_dist.get(v, 0.01)
            joint = p_fault * likelihood
            p_outcomes[v] += joint
            p_fault_given_outcome[v][fault.fault_id] = joint

    # 3. Expected conditional entropy across all test outcomes
    expected_conditional_entropy = 0.0
    total_marginal = sum(p_outcomes.values())
    if total_marginal <= 0:
        return 0.0

    for v, p_v in p_outcomes.items():
        if p_v <= 1e-9:
            continue
        p_v_normalized = p_v / total_marginal
        # Normalize posterior given outcome v
        posteriors_given_v = [
            p_fault_given_outcome[v][f.fault_id] / p_v
            for f in faults
        ]
        h_v = calculate_entropy(posteriors_given_v)
        expected_conditional_entropy += p_v_normalized * h_v

    # 4. EIG = H(Prior) - E[H(Posterior)]
    eig = base_entropy - expected_conditional_entropy
    return max(0.0, eig)


def select_best_follow_up_question(
    ranked_hypotheses: List[Dict[str, Any]],
    current_observations: Dict[str, Any],
    fault_model: FaultModel,
    confidence_gap: float = 0.0,
    is_ambiguous: bool = False,
    insufficient_evidence: bool = False,
    min_eig_threshold: float = 0.05,
    max_candidates_to_consider: int = 5,
) -> Optional[QuestionCandidate]:
    """
    Selects the optimal follow-up diagnostic question by calculating EIG over competing candidates.
    Returns None if:
    - Case is unambiguous (confidence is high, confidence gap is wide)
    - All distinguishing tests are already observed
    - Maximum EIG is below min_eig_threshold
    """
    # Stopping criterion 1: If top hypothesis has high confidence and clear gap, do not question
    if not is_ambiguous and not insufficient_evidence and confidence_gap >= 0.25:
        return None

    if not ranked_hypotheses:
        return None

    # Consider top competing faults
    top_hyps = ranked_hypotheses[:max_candidates_to_consider]
    top_fault_ids = [h["fault_id"] for h in top_hyps]
    fault_objects = [fault_model.get_fault(fid) for fid in top_fault_ids if fault_model.get_fault(fid)]
    if not fault_objects:
        return None

    posterior_dict = {h["fault_id"]: h.get("posterior_prob", 0.0) for h in top_hyps}

    # Already observed symptoms
    observed_symptoms: Set[str] = set(current_observations.get("symptoms", {}).keys())

    # Candidate tests: all tests defined in TEST_QUESTION_METADATA that are NOT yet observed
    available_tests = [
        test_id for test_id in TEST_QUESTION_METADATA.keys()
        if test_id not in observed_symptoms
    ]

    if not available_tests:
        return None

    # Score each available test by EIG
    scored_tests: List[QuestionCandidate] = []
    for test_id in available_tests:
        eig = compute_eig_for_test(test_id, fault_objects, posterior_dict)
        if eig >= min_eig_threshold:
            meta = TEST_QUESTION_METADATA[test_id]
            # Identify which top candidates this test separates
            top_names = [h.get("fault_name", h["fault_id"]) for h in top_hyps[:2]]
            candidate = QuestionCandidate(
                test_id=test_id,
                test_name=meta["test_name"],
                question_text=meta["question_text"],
                possible_answers=meta["possible_answers"],
                eig_score=round(eig, 4),
                rationale=meta["rationale"],
                discriminates_between=top_names,
            )
            scored_tests.append(candidate)

    if not scored_tests:
        return None

    # Sort descending by EIG
    scored_tests.sort(key=lambda q: q.eig_score, reverse=True)
    return scored_tests[0]
