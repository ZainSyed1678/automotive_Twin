"""
Quick demonstration runner for M1 Fault Model.
Runs a simulation and three diagnostic scoring scenarios.
"""

from model import FaultModel, Observations


def print_result(title: str, result):
    print("=" * 70)
    print(f"CASE: {title}")
    print("=" * 70)
    print(f"Top Diagnosis       : {result.top_fault_id}")
    print(f"Confidence Gap      : {result.confidence_gap:.4f}")
    print(f"Is Ambiguous        : {result.is_ambiguous}")
    print(f"Insufficient Evidence: {result.insufficient_evidence}")
    print("\nTop 3 Hypotheses:")
    for i, hyp in enumerate(result.ranked_hypotheses[:3], start=1):
        print(f"  {i}. {hyp.fault_name} ({hyp.fault_id})")
        print(f"     Probability: {hyp.posterior_prob * 100:.1f}% | Log-Likelihood: {hyp.log_likelihood:.2f}")
        if hyp.evidence_matches:
            print(f"     Matches: {', '.join(hyp.evidence_matches[:2])}")
        if hyp.evidence_contradictions:
            print(f"     Contradictions: {', '.join(hyp.evidence_contradictions[:2])}")

    print("\nRecommended Next Tests:")
    for test in result.suggested_tests[:3]:
        print(f"  - {test}")
    print()


def main():
    model = FaultModel()
    print("Loaded Fault Catalogue:")
    print(f"  Components: {len(model.list_components())}")
    print(f"  Faults    : {len(model.list_faults())}\n")

    # 1. Simulation demo
    print("Simulating 'intake_vacuum_leak':")
    sim = model.simulate_fault("intake_vacuum_leak")
    print(f"  Component: {sim.component_id}")
    print(f"  Expected DTCs: {sim.expected_dtcs}")
    print(f"  Dominant Symptoms: {sim.dominant_symptoms}\n")

    # 2. Ambiguous Case (Only lean DTC P0171, no physical inspection yet)
    res_ambiguous = model.score_hypotheses(Observations(dtc_codes=["P0171"]))
    print_result("Ambiguous Initial Intake (DTC P0171 Only)", res_ambiguous)

    # 3. Disambiguated Case: Smoke test / vacuum symptoms observed
    res_leak = model.score_hypotheses(
        Observations(
            dtc_codes=["P0171"],
            symptoms={
                "fuel_pressure": "normal",
                "fuel_trim_at_high_rpm": "normalizes",
                "audible_hiss": "present"
            }
        )
    )
    print_result("Resolved Case (P0171 + Normal Fuel Pressure + High RPM Trims Normalize)", res_leak)

    # 4. Disambiguated Case: Fuel delivery failure observed
    res_pump = model.score_hypotheses(
        Observations(
            dtc_codes=["P0171", "P0087"],
            symptoms={
                "fuel_pressure": "low",
                "fuel_trim_at_high_rpm": "worsens",
                "engine_load_behavior": "loss_of_power"
            }
        )
    )
    print_result("Resolved Case (P0171 + P0087 + Low Pressure + High RPM Trims Worsen)", res_pump)


if __name__ == "__main__":
    main()
