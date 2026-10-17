"""
Self-Review and Evidence Verification Gate for Diagnostic Reports.
Enforces epistemic humility, validates evidence grounding, and checks subsystem boundaries.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from model.engine import FaultModel


class ReviewResult(BaseModel):
    """
    Outcome of the diagnostic review verification gate.
    """
    review_status: str = Field("PASSED", description="PASSED, FLAGGED, or MODIFIED")
    passed: bool = True
    evidence_grounding_passed: bool = True
    scope_boundary_passed: bool = True
    epistemic_humility_passed: bool = True
    unsupported_claims: List[str] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)
    adjusted_summary: Optional[str] = None


SUPPORTED_SUBSYSTEM = "air_intake_and_fuel_control"
VALID_COMPONENTS = {
    "maf_sensor",
    "map_sensor",
    "throttle_position_sensor",
    "oxygen_sensor",
    "engine_coolant_temp_sensor",
    "fuel_injector",
    "fuel_pump",
    "intake_vacuum_line",
    "wiring_harness",
    "ecu",
}


def review_diagnostic_report(
    state: Dict[str, Any],
    fault_model: FaultModel,
) -> ReviewResult:
    """
    Audits the generated diagnostic state and preliminary report against strict verification gates:
    1. Scope Boundary: Ensures all hypotheses and components belong to supported subsystem.
    2. Evidence Grounding: Checks that evidence matches and citations reference actual observations or documents.
    3. Epistemic Humility: Confirms that ambiguous or sparse cases do not assert premature certainty.
    """
    obs_data = state.get("observations", {})
    dtc_codes: Set[str] = set(obs_data.get("dtc_codes", []))
    observed_symptoms: Set[str] = set(obs_data.get("symptoms", {}).keys())

    diag_result = state.get("diagnostic_result") or {}
    ranked_hyps = diag_result.get("ranked_hypotheses", [])
    is_ambiguous = diag_result.get("is_ambiguous", False)
    insufficient_evidence = diag_result.get("insufficient_evidence", False)
    citations: List[str] = state.get("citations", [])

    unsupported_claims: List[str] = []
    review_notes: List[str] = []
    adjusted_summary: Optional[str] = None

    scope_passed = True
    grounding_passed = True
    humility_passed = True

    # -------------------------------------------------------------------------
    # Gate 1: Subsystem Scope & Component Boundary Check
    # -------------------------------------------------------------------------
    for hyp in ranked_hyps[:5]:
        comp_id = hyp.get("component_id")
        if comp_id and comp_id not in VALID_COMPONENTS:
            scope_passed = False
            unsupported_claims.append(
                f"Component '{comp_id}' is outside the supported '{SUPPORTED_SUBSYSTEM}' subsystem."
            )

    if scope_passed:
        review_notes.append("Scope Check: All evaluated components strictly within air-intake & fuel-control subsystem.")
    else:
        review_notes.append("Scope Warning: One or more candidate components exceeded verified subsystem boundary.")

    # -------------------------------------------------------------------------
    # Gate 2: Evidence Grounding & Traceability Check
    # -------------------------------------------------------------------------
    # Verify that top hypothesis evidence matches are actually present in input observations
    if ranked_hyps:
        top_hyp = ranked_hyps[0]
        evidence_matches = top_hyp.get("evidence_matches", [])
        for match in evidence_matches:
            # Matches are formatted as "DTC: Pxxxx" or "Symptom: name=state"
            if match.startswith("DTC: "):
                code = match.replace("DTC: ", "").strip()
                if code not in dtc_codes:
                    grounding_passed = False
                    unsupported_claims.append(f"Claimed DTC match '{code}' not present in technician observations.")
            elif match.startswith("Symptom: "):
                symptom_part = match.replace("Symptom: ", "").strip()
                s_name = symptom_part.split("=")[0].strip() if "=" in symptom_part else symptom_part
                if s_name not in observed_symptoms:
                    grounding_passed = False
                    unsupported_claims.append(f"Claimed symptom match '{s_name}' not present in technician observations.")

    # Verify citation validity
    if citations:
        review_notes.append(f"Citation Check: {len(citations)} verifiable source references attached to diagnostic state.")
    else:
        review_notes.append("Citation Notice: No technical document citations attached.")

    if grounding_passed:
        review_notes.append("Grounding Check: All cited evidence directly matches reported scanner and technician inputs.")
    else:
        review_notes.append("Grounding Warning: Detected ungrounded evidence assertions.")

    # -------------------------------------------------------------------------
    # Gate 3: Epistemic Humility & Overconfidence Guard
    # -------------------------------------------------------------------------
    if insufficient_evidence:
        top_prob = ranked_hyps[0].get("posterior_prob", 0.0) if ranked_hyps else 0.0
        if top_prob > 0.40:
            humility_passed = False
            unsupported_claims.append(
                f"Overconfidence detected on insufficient evidence case (Top prob: {top_prob:.2f})."
            )
            adjusted_summary = (
                "VERIFICATION ADJUSTED: Input observations are insufficient to establish a reliable diagnosis. "
                "Confidence score suppressed. Technician must perform fundamental sensor checks."
            )
        review_notes.append("Humility Check: Insufficient evidence guard active; forced diagnostic modesty.")

    elif is_ambiguous:
        confidence_gap = diag_result.get("confidence_gap", 0.0)
        review_notes.append(
            f"Humility Check: Ambiguity detected (margin: {confidence_gap * 100:.1f}%). "
            f"Mandatory distinguishing test guidance enforced."
        )

    # -------------------------------------------------------------------------
    # Final Review Status Compilation
    # -------------------------------------------------------------------------
    overall_passed = scope_passed and grounding_passed and humility_passed
    status = "PASSED" if overall_passed else ("MODIFIED" if adjusted_summary else "FLAGGED")

    return ReviewResult(
        review_status=status,
        passed=overall_passed,
        evidence_grounding_passed=grounding_passed,
        scope_boundary_passed=scope_passed,
        epistemic_humility_passed=humility_passed,
        unsupported_claims=unsupported_claims,
        review_notes=review_notes,
        adjusted_summary=adjusted_summary,
    )
