"""
Automated Repair Action Plan and Cost Estimator Engine.
Synthesizes Bayesian diagnostic results, OEM repair specifications,
labor book times, and safety protocols into an actionable repair plan.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from agents.state import DiagnosticReport
from .schema import (
    RepairPlan,
    PartEstimate,
    LaborEstimate,
    SafetyWarning,
    RepairStep,
    DriveCycleStep,
    VerificationProtocol,
)

CATALOGUE_PATH = Path(__file__).parent / "catalogue.yaml"


def load_repair_catalogue() -> Dict[str, Any]:
    """Loads the repair procedure database with robust path resolution."""
    candidate_paths = [
        CATALOGUE_PATH,
        Path(__file__).parent / "catalogue.yaml",
        Path.cwd() / "repair" / "catalogue.yaml",
        Path("/home/user/app/repair/catalogue.yaml"),
    ]
    for p in candidate_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("repairs", {})
    return {}


def generate_repair_plan(
    diagnosis: DiagnosticReport,
    hourly_rate_usd: float = 110.0,
) -> RepairPlan:
    """
    Generates a complete, actionable repair action plan, parts/labor cost estimate,
    SOP steps, and verification drive cycle for a diagnosed fault.
    """
    catalogue = load_repair_catalogue()
    fault_id = diagnosis.top_fault_id or "default_generic"
    entry = catalogue.get(fault_id) or catalogue.get("default_generic", {})

    urgency = entry.get("urgency_level", "MODERATE_SCHEDULE_SERVICE")
    component_id = entry.get("component_id", diagnosis.component_id or "unknown")

    # 1. Build Part Estimates
    parts = []
    total_parts = 0.0
    for p in entry.get("parts", []):
        qty = p.get("quantity", 1)
        unit = float(p.get("unit_cost_usd", 0.0))
        tot = round(qty * unit, 2)
        total_parts += tot
        parts.append(
            PartEstimate(
                part_number=p.get("part_number", "OEM-REPLACE"),
                description=p.get("description", "Replacement Component"),
                quantity=qty,
                unit_cost_usd=unit,
                total_cost_usd=tot,
                category=p.get("category", "oem"),
            )
        )

    # 2. Build Labor Estimates
    labor_data = entry.get("labor", {})
    book_hrs = float(labor_data.get("book_hours", 1.5))
    labor_cost = round(book_hrs * hourly_rate_usd, 2)
    labor_estimates = [
        LaborEstimate(
            operation=labor_data.get("operation", "Service / Replace Component"),
            book_hours=book_hrs,
            hourly_rate_usd=hourly_rate_usd,
            labor_cost_usd=labor_cost,
            technician_skill=labor_data.get("technician_skill", "ASE_A8_ENGINE_PERFORMANCE"),
        )
    ]

    # 3. Build Safety Warnings
    warnings = []
    is_ev = (
        "prius" in (diagnosis.vehicle_id or "").lower()
        or "tesla" in (diagnosis.vehicle_id or "").lower()
        or "bev" in (diagnosis.vehicle_id or "").lower()
        or "hybrid" in (diagnosis.vehicle_id or "").lower()
        or fault_id in [
            "bms_cell_voltage_imbalance",
            "high_voltage_isolation_loss",
            "battery_cooling_pump_failure",
            "inverter_phase_loss",
            "traction_motor_resolver_fault",
        ]
    )

    for w in entry.get("safety_warnings", []):
        warnings.append(
            SafetyWarning(
                severity=w.get("severity", "WARNING"),
                standard=w.get("standard", "OSHA Standard"),
                instruction=w.get("instruction", ""),
            )
        )

    # Automatic High Voltage safety injection if servicing EV powertrain without explicit warning
    if is_ev and not any(w.severity == "DANGER_HIGH_VOLTAGE" for w in warnings):
        warnings.insert(
            0,
            SafetyWarning(
                severity="DANGER_HIGH_VOLTAGE",
                standard="NFPA 70E High-Voltage Safety Protocol",
                instruction="High-voltage circuit present (>60V DC / >30V AC). De-energize high voltage via Service Disconnect, verify 0.0V using 1000V CAT III/IV meter, and wear Class 0 rated insulated gloves.",
            ),
        )

    # 4. Build Procedure Steps
    steps = []
    for s in entry.get("procedure_steps", []):
        steps.append(
            RepairStep(
                step_number=s.get("step_number", len(steps) + 1),
                phase=s.get("phase", "COMPONENT_SERVICE"),
                instruction=s.get("instruction", ""),
                torque_spec=s.get("torque_spec"),
                required_tools=s.get("required_tools", []),
            )
        )

    # 5. Build Verification Protocol
    vp_data = entry.get("verification_protocol", {})
    dc_steps = []
    for d in vp_data.get("drive_cycle_steps", []):
        dc_steps.append(
            DriveCycleStep(
                step_number=d.get("step_number", len(dc_steps) + 1),
                phase_name=d.get("phase_name", "Verification Run"),
                condition=d.get("condition", "Normal driving condition"),
                duration_s=float(d.get("duration_s", 300.0)),
                target_speed_kmh=float(d.get("target_speed_kmh", 50.0)),
                target_rpm=float(d.get("target_rpm", 2000.0)),
                monitor_readiness_check=d.get("monitor_readiness_check", "Pass"),
            )
        )

    verification = VerificationProtocol(
        monitor_target=vp_data.get("monitor_target", "OBD-II Readiness Monitors"),
        drive_cycle_steps=dc_steps,
        pass_criteria=vp_data.get("pass_criteria", ["No active DTCs", "Engine runs smooth"]),
        reset_instructions=vp_data.get("reset_instructions", ["Clear DTCs using scan tool"]),
    )

    total_est = round(total_parts + labor_cost, 2)

    return RepairPlan(
        fault_id=fault_id,
        fault_name=diagnosis.top_fault_name or fault_id.replace("_", " ").title(),
        component_id=component_id,
        vehicle_id=diagnosis.vehicle_id or "generic_petrol",
        urgency_level=urgency,
        parts=parts,
        labor=labor_estimates,
        total_parts_cost_usd=round(total_parts, 2),
        total_labor_cost_usd=round(labor_cost, 2),
        total_estimated_cost_usd=total_est,
        safety_warnings=warnings,
        procedure_steps=steps,
        verification_protocol=verification,
    )
