"""
Tests for Phase 6: Automated Repair Action Plan, Labor/Parts Cost Estimator & Post-Repair Verification.
"""

import pytest
from agents import DiagnosticReport
from repair import (
    generate_repair_plan,
    verify_post_repair_drive_cycle,
    RepairPlan,
    load_repair_catalogue,
)
from stream import generate_telemetry_stream


def test_load_repair_catalogue():
    """Verify repair catalogue loads and contains essential failure modes."""
    cat = load_repair_catalogue()
    assert len(cat) >= 10
    assert "intake_vacuum_leak" in cat
    assert "bms_cell_voltage_imbalance" in cat
    assert "catalyst_efficiency_degraded" in cat


def test_ice_vacuum_leak_repair_plan():
    """Verify detailed repair plan, parts, labor calculation, and torque specs for vacuum leak."""
    diag = DiagnosticReport(
        top_fault_id="intake_vacuum_leak",
        top_fault_name="Intake Manifold / Vacuum Line Leak",
        component_id="intake_vacuum_line",
        vehicle_id="vw_golf_14_tsi",
        confidence_score=0.91,
    )
    plan = generate_repair_plan(diag, hourly_rate_usd=120.0)

    assert isinstance(plan, RepairPlan)
    assert plan.fault_id == "intake_vacuum_leak"
    assert plan.urgency_level == "HIGH_SERVICE_SOON"
    assert len(plan.parts) >= 1
    assert plan.labor[0].book_hours == 2.2
    assert plan.labor[0].labor_cost_usd == round(2.2 * 120.0, 2)
    assert plan.total_estimated_cost_usd == round(plan.total_parts_cost_usd + plan.total_labor_cost_usd, 2)
    assert any("10 Nm" in (s.torque_spec or "") for s in plan.procedure_steps)
    assert len(plan.verification_protocol.drive_cycle_steps) >= 1


def test_ev_high_voltage_safety_and_specialist_skills():
    """Verify NFPA 70E High Voltage safety warnings and L3 specialist requirements for EV faults."""
    diag = DiagnosticReport(
        top_fault_id="bms_cell_voltage_imbalance",
        top_fault_name="HV Battery Cell Imbalance",
        component_id="battery_management_system",
        vehicle_id="toyota_prius_18_hybrid",
        confidence_score=0.95,
    )
    plan = generate_repair_plan(diag, hourly_rate_usd=135.0)

    assert plan.urgency_level == "CRITICAL_IMMEDIATE_STOP"
    assert plan.labor[0].technician_skill == "ASE_L3_HYBRID_EV_SPECIALIST"
    
    # Check High Voltage safety warning
    hv_warnings = [w for w in plan.safety_warnings if w.severity == "DANGER_HIGH_VOLTAGE"]
    assert len(hv_warnings) >= 1
    assert "NFPA 70E" in hv_warnings[0].standard
    assert "1000V" in hv_warnings[0].instruction
    
    # Check verification protocol
    assert any("Regenerative" in s.phase_name for s in plan.verification_protocol.drive_cycle_steps)


def test_exhaust_catalyst_repair_plan():
    """Verify catalytic converter repair plan and EPA Federal drive cycle."""
    diag = DiagnosticReport(
        top_fault_id="catalyst_efficiency_degraded",
        top_fault_name="Catalytic Converter Efficiency Low",
        component_id="catalytic_converter",
        vehicle_id="hyundai_i20_12_mpi",
        confidence_score=0.88,
    )
    plan = generate_repair_plan(diag, hourly_rate_usd=100.0)

    assert plan.total_parts_cost_usd >= 500.0  # Precious metals catalytic converter
    assert any("EPA" in plan.verification_protocol.monitor_target for plan in [plan])
    assert any("Downstream O2" in c for c in plan.verification_protocol.pass_criteria)


def test_custom_hourly_shop_rate():
    """Verify custom hourly shop rate scales labor cost and total estimate precisely."""
    diag = DiagnosticReport(top_fault_id="spark_plug_fouling", vehicle_id="generic_petrol")
    plan_cheap = generate_repair_plan(diag, hourly_rate_usd=80.0)
    plan_expensive = generate_repair_plan(diag, hourly_rate_usd=160.0)

    assert plan_cheap.labor[0].labor_cost_usd == 80.0 * plan_cheap.labor[0].book_hours
    assert plan_expensive.labor[0].labor_cost_usd == 160.0 * plan_expensive.labor[0].book_hours
    assert plan_expensive.total_estimated_cost_usd > plan_cheap.total_estimated_cost_usd


def test_post_repair_drive_cycle_verification_success():
    """Verify that a successful post-repair drive cycle passes verification checks."""
    # Healthy stream after repairing vacuum leak
    healthy_stream = generate_telemetry_stream(vehicle_id="vw_golf_14_tsi", fault_type="healthy", duration_s=20.0)
    result = verify_post_repair_drive_cycle(fault_id="intake_vacuum_leak", stream=healthy_stream)

    assert result["verified"] is True
    assert result["readiness_status"] == "PASSED_VERIFIED"
    assert any("Fuel System Monitor: READY" in m for m in result["monitors_ready"])
    assert len(result["failed_checks"]) == 0


def test_post_repair_drive_cycle_verification_failure():
    """Verify that an unresolved fault stream triggers failure in post-repair verification."""
    # Fault still present in stream
    leaking_stream = generate_telemetry_stream(vehicle_id="vw_golf_14_tsi", fault_type="vacuum_leak", duration_s=20.0)
    result = verify_post_repair_drive_cycle(fault_id="intake_vacuum_leak", stream=leaking_stream)

    assert result["verified"] is False
    assert result["readiness_status"] == "FAILED_UNRESOLVED"
    assert len(result["failed_checks"]) >= 1
    assert any("Idle LTFT remains elevated" in f for f in result["failed_checks"])


def test_post_repair_ev_inverter_verification():
    """Verify post-repair verification for EV inverter thermal pump replacement."""
    healthy_stream = generate_telemetry_stream(vehicle_id="toyota_prius_18_hybrid", fault_type="healthy", duration_s=20.0)
    result = verify_post_repair_drive_cycle(fault_id="battery_cooling_pump_failure", stream=healthy_stream)

    assert result["verified"] is True
    assert any("Inverter Thermal Monitor: READY" in m for m in result["monitors_ready"])


def test_generic_fallback_repair_plan():
    """Verify unrecognized or uncatalogued fault ID falls back safely to default plan."""
    diag = DiagnosticReport(top_fault_id="unusual_novel_sensor_short", vehicle_id="generic_petrol")
    plan = generate_repair_plan(diag)

    assert isinstance(plan, RepairPlan)
    assert plan.total_estimated_cost_usd > 0.0
    assert len(plan.procedure_steps) >= 1
