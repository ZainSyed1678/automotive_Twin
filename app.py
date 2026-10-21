"""
Automotive Digital Twin Diagnostic Station - Gradio Interface.
Deployment-ready entrypoint for Hugging Face Spaces (Gradio SDK).
Supports 4 Subsystems: Air-Intake & Fuel, Ignition & Combustion, Exhaust & Emissions, EV & Hybrid Powertrain.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# ZeroGPU compatibility for Hugging Face Spaces
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None:
                return fn
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from agents.workflow import run_diagnosis
from model.schema import Observations
from model.vehicle_registry import vehicle_registry
from stream import generate_telemetry_stream, analyze_telemetry_stream
from repair import generate_repair_plan, verify_post_repair_drive_cycle, load_repair_catalogue


VEHICLE_OPTIONS = {
    "Generic 4-Stroke Petrol Engine (MPI Baseline)": "generic_petrol",
    "Hyundai i20 1.2L Kappa II MPI (2014-2020)": "hyundai_i20_kappa_12",
    "Volkswagen Golf 1.4L TSI EA211 GDI (2013-2020)": "vw_golf_14_tsi",
    "Toyota Prius 1.8L Hybrid Synergy Drive (2015-2022)": "toyota_prius_18_hybrid",
    "Tesla Model 3 Long Range Dual Motor BEV (2017-2023)": "tesla_model_3_bev",
}

PRESET_SCENARIOS = {
    "-- Select a Preset --": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": [],
        "idle_quality": "Unobserved",
        "fuel_pressure": "Unobserved",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "ignition_coil_swap": "Unobserved",
        "downstream_o2": "Unobserved",
        "cell_voltage_delta": "Unobserved",
        "inverter_temp": "Unobserved",
        "isolation_resistance": "Unobserved",
        "notes": "",
    },
    "Scenario 1: MAF Contamination (P0101, Sluggish)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0101"],
        "idle_quality": "Unobserved",
        "fuel_pressure": "Unobserved",
        "fuel_trim_ltft": "high_positive",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "underreporting",
        "audible_hiss": "Unobserved",
        "notes": "Vehicle hesitates on tip-in throttle. MAF sensor hot-wire film suspected.",
    },
    "Scenario 2: Ambiguous Lean (P0171 - Follow-up Needed)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0171"],
        "idle_quality": "Unobserved",
        "fuel_pressure": "Unobserved",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "notes": "Generic system lean code. Multiple plausible root causes.",
    },
    "Scenario 3: Weak Fuel Pump (P0171, Low Pressure)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0171"],
        "idle_quality": "Unobserved",
        "fuel_pressure": "low",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "worsens",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "notes": "Fuel rail pressure drops below 2.0 bar under load.",
    },
    "Scenario 4: Intake Vacuum Leak (P0171, P0507, Hissing)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0171", "P0507"],
        "idle_quality": "rough",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "high_positive",
        "fuel_trim_at_high_rpm": "normalizes",
        "maf_reading": "Unobserved",
        "audible_hiss": "present",
        "notes": "Audible whistling around intake manifold runner gasket.",
    },
    "Scenario 5: VW Golf EA211 Direct Injection Carbon Fouling (P0300)": {
        "vehicle": "Volkswagen Golf 1.4L TSI EA211 GDI (2013-2020)",
        "dtcs": ["P0300"],
        "idle_quality": "rough",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "high_positive",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "notes": "VW 1.4 TSI EA211 rough idle on cold start. Intake valve carbon buildup suspected.",
    },
    "Scenario 6: Catalytic Converter Degradation (P0420, Downstream O2 Active)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0420"],
        "idle_quality": "normal",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "normal",
        "downstream_o2": "active_mirroring_upstream",
        "notes": "Catalyst efficiency low. Downstream O2 sensor mirrors upstream sensor oscillation.",
    },
    "Scenario 7: Toyota Prius Inverter Coolant Pump Failure (P0A93, Overheating)": {
        "vehicle": "Toyota Prius 1.8L Hybrid Synergy Drive (2015-2022)",
        "dtcs": ["P0A93"],
        "idle_quality": "normal",
        "inverter_temp": "abnormally_high",
        "battery_cooling_flow": "no_circulation",
        "hybrid_system_warning": "present",
        "notes": "Prius master warning triangle illuminated. Inverter coolant reservoir shows zero fluid movement.",
    },
    "Scenario 8: Tesla Model 3 High-Voltage Isolation Loss (P0AA6, Warning Light)": {
        "vehicle": "Tesla Model 3 Long Range Dual Motor BEV (2017-2023)",
        "dtcs": ["P0AA6"],
        "isolation_resistance": "loss_of_isolation",
        "high_voltage_warning": "warning_present",
        "notes": "Tesla UI displays 'Electrical system issue - Vehicle may not restart'. Chassis insulation degradation.",
    },
    "Scenario 9: Toyota Prius Battery Cell Imbalance (P0A7F, Derated Acceleration)": {
        "vehicle": "Toyota Prius 1.8L Hybrid Synergy Drive (2015-2022)",
        "dtcs": ["P0A7F"],
        "cell_voltage_delta": "excessive_imbalance",
        "ev_acceleration": "derated",
        "hybrid_system_warning": "present",
        "notes": "Prius battery block delta exceeds 120 mV under brisk acceleration.",
    },
}


def load_preset(preset_name: str):
    data = PRESET_SCENARIOS.get(preset_name, PRESET_SCENARIOS["-- Select a Preset --"])
    return (
        data.get("vehicle", "Generic 4-Stroke Petrol Engine (MPI Baseline)"),
        data.get("dtcs", []),
        data.get("idle_quality", "Unobserved"),
        data.get("fuel_pressure", "Unobserved"),
        data.get("fuel_trim_ltft", "Unobserved"),
        data.get("fuel_trim_at_high_rpm", "Unobserved"),
        data.get("maf_reading", "Unobserved"),
        data.get("audible_hiss", "Unobserved"),
        data.get("ignition_coil_swap", "Unobserved"),
        data.get("downstream_o2", "Unobserved"),
        data.get("cell_voltage_delta", "Unobserved"),
        data.get("inverter_temp", "Unobserved"),
        data.get("isolation_resistance", "Unobserved"),
        data.get("notes", ""),
    )


@spaces.GPU
def diagnose(
    vehicle_name: str,
    dtcs: List[str],
    idle_quality: str,
    fuel_pressure: str,
    fuel_trim_ltft: str,
    fuel_trim_at_high_rpm: str,
    maf_reading: str,
    audible_hiss: str,
    ignition_coil_swap: str,
    downstream_o2: str,
    cell_voltage_delta: str,
    inverter_temp: str,
    isolation_resistance: str,
    notes: str,
    shop_rate: float = 110.0,
):
    vehicle_id = VEHICLE_OPTIONS.get(vehicle_name, "generic_petrol")

    # Assemble symptom dict
    symptoms = {}
    for val, key in [
        (idle_quality, "idle_quality"),
        (fuel_pressure, "fuel_pressure"),
        (fuel_trim_ltft, "fuel_trim_ltft"),
        (fuel_trim_at_high_rpm, "fuel_trim_at_high_rpm"),
        (maf_reading, "maf_reading"),
        (audible_hiss, "audible_hiss"),
        (ignition_coil_swap, "ignition_coil_swap"),
        (downstream_o2, "downstream_o2_switching"),
        (cell_voltage_delta, "cell_voltage_delta"),
        (inverter_temp, "inverter_temp"),
        (isolation_resistance, "chassis_isolation_resistance"),
    ]:
        if val and val != "Unobserved":
            symptoms[key] = val

    obs = Observations(
        dtc_codes=dtcs or [],
        symptoms=symptoms,
        notes=notes or None,
        vehicle_id=vehicle_id,
    )

    report = run_diagnosis(obs)

    # 1. Status banner
    if report.review_status == "PASSED":
        badge = "### 🛡️ **Self-Review Gate: PASSED** (Grounding & Subsystem Scope Verified)"
    else:
        badge = f"### ⚠️ **Self-Review Gate: {report.review_status}**"

    # 2. Executive summary
    cascading_str = ""
    if report.ranked_hypotheses and report.ranked_hypotheses[0].get("cascading_risks"):
        risks = report.ranked_hypotheses[0]["cascading_risks"]
        risk_bullets = "\n".join([f"- ⚠️ **Cross-Subsystem Hazard:** {r}" for r in risks])
        cascading_str = f"\n\n### ⚡ Cross-Subsystem Cascading Effects\n{risk_bullets}\n"

    summary_md = f"""
{badge}

**Vehicle Platform:** `{report.vehicle_name or vehicle_id}`  
**Top Hypothesis:** `{report.top_fault_name or 'Indeterminate'}`  
**Posterior Confidence:** `{report.confidence_score * 100:.1f}%` | **Margin Gap:** `{report.confidence_gap * 100:.1f}%`  
**Diagnostic Status:** `{'Insufficient Evidence' if report.insufficient_evidence else ('Ambiguous' if report.is_ambiguous else 'High Confidence')}`

> **Summary:** {report.executive_summary}
{cascading_str}
"""

    # 3. Ranked hypotheses table
    table_rows = []
    for idx, h in enumerate(report.ranked_hypotheses[:5], 1):
        table_rows.append({
            "Rank": idx,
            "Failure Mode": h.get("fault_name", ""),
            "Subsystem": h.get("subsystem", "air_intake_and_fuel_control"),
            "Component": h.get("component_id", ""),
            "Posterior Probability": f"{h.get('posterior_prob', 0.0) * 100:.1f}%",
            "Weight of Evidence (Score)": round(h.get("log_likelihood", 0.0), 2),
        })
    hyp_df = pd.DataFrame(table_rows)

    # 4. Follow-up question (EIG)
    if report.follow_up_question:
        q = report.follow_up_question
        q_md = f"""
### ❓ Targeted Follow-up Inspection (EIG: `{q['eig_score']:.3f} bits`)
**Inspection Procedure:** {q['question_text']}  
*Options:* `{', '.join(q['possible_answers'])}`  
*Rationale:* {q['rationale']}
"""
    else:
        q_md = "*(No follow-up test needed — diagnosis meets confidence stopping threshold)*"

    # 5. Citations & evidence
    citations_list = [f"- `{c}`" for c in report.citations]
    cit_md = "### 📚 Verifiable Technical Citations\n" + ("\n".join(citations_list) if citations_list else "_No citations attached_")

    matches = [f"- ✅ `{m}`" for m in report.supporting_evidence]
    contra = [f"- ❌ `{c}`" for c in report.contradictory_evidence]
    ev_md = "### 🔍 Grounded Evidence Breakdown\n**Matches:**\n" + ("\n".join(matches) if matches else "_None_") + "\n\n**Contradictions:**\n" + ("\n".join(contra) if contra else "_None_")

    # 6. Automated Repair Plan
    repair_plan = generate_repair_plan(report, hourly_rate_usd=shop_rate)
    parts_lines = [f"- `{p.part_number}`: {p.description} (Qty: {p.quantity}) — **${p.total_cost_usd:.2f}** [{p.category.upper()}]" for p in repair_plan.parts]
    labor_lines = [f"- {l.operation}: {l.book_hours} hrs @ ${l.hourly_rate_usd:.2f}/hr — **${l.labor_cost_usd:.2f}** ({l.technician_skill})" for l in repair_plan.labor]
    safety_lines = [f"- **{w.severity}** ({w.standard}): {w.instruction}" for w in repair_plan.safety_warnings]
    steps_lines = [f"{s.step_number}. **[{s.phase}]** {s.instruction}" + (f" *(Torque: {s.torque_spec})*" if s.torque_spec else "") for s in repair_plan.procedure_steps]

    repair_md = f"""
### 🛠️ Work Order & Repair Estimate: `{repair_plan.fault_name}`
**Urgency:** `{repair_plan.urgency_level}` | **Total Estimate:** `${repair_plan.total_estimated_cost_usd:.2f}` (Parts: `${repair_plan.total_parts_cost_usd:.2f}` | Labor: `${repair_plan.total_labor_cost_usd:.2f}`)

#### ⚠️ Safety Precautions
{chr(10).join(safety_lines) if safety_lines else '_Standard shop safety applies._'}

#### 📦 Required Parts
{chr(10).join(parts_lines) if parts_lines else '_No replacement parts specified._'}

#### ⏱️ Flat-Rate Labor
{chr(10).join(labor_lines) if labor_lines else '_Inspection only._'}

#### 📋 Standard Operating Procedure
{chr(10).join(steps_lines)}

#### 🔄 Post-Repair Verification Drive Cycle
**Target Monitor:** `{repair_plan.verification_protocol.monitor_target}`  
**Pass Criteria:**  
{chr(10).join(['- ' + c for c in repair_plan.verification_protocol.pass_criteria])}
"""

    return summary_md, hyp_df, q_md, cit_md, ev_md, repair_md


def simulate_and_diagnose_stream(vehicle_choice: str, fault_sim: str, duration_s: float):
    """Simulates 10Hz OBD-II stream, analyzes waveforms, and feeds into diagnosis."""
    v_id = VEHICLE_OPTIONS.get(vehicle_choice, "generic_petrol")
    stream = generate_telemetry_stream(vehicle_id=v_id, fault_type=fault_sim, duration_s=duration_s)
    analysis = analyze_telemetry_stream(stream)
    report = run_diagnosis(analysis.extracted_observations)

    # Format stream metrics
    metrics_md = f"""
### 📈 10Hz Time-Series Telemetry Processing
- **Total Frames Ingested:** `{analysis.total_frames}` frames across `{analysis.duration_s:.1f}s`
- **O2 Switching Frequency:** `{analysis.signal_metrics.get('o2_switching_freq_hz', 0.0):.2f} Hz`
- **Downstream O2 Variation (Std):** `{analysis.signal_metrics.get('downstream_o2_std_v', 0.0):.3f} V`
- **LTFT Idle Mean:** `{analysis.signal_metrics.get('ltft_idle_mean_pct', 0.0):+.1f}%` | **Cruise Mean:** `{analysis.signal_metrics.get('ltft_cruise_mean_pct', 0.0):+.1f}%`
"""
    if "max_cell_delta_mv" in analysis.signal_metrics:
        metrics_md += f"- **HV Battery Max Cell Delta:** `{analysis.signal_metrics['max_cell_delta_mv']:.1f} mV`\n"
    if "max_inverter_temp_c" in analysis.signal_metrics:
        metrics_md += f"- **Inverter Peak Temperature:** `{analysis.signal_metrics['max_inverter_temp_c']:.1f} °C`\n"

    anomalies_list = [f"- ⚠️ **Anomaly Detected:** {a}" for a in analysis.detected_anomalies]
    anomaly_md = "#### Physical Waveform Anomalies\n" + ("\n".join(anomalies_list) if anomalies_list else "_Telemetry waveforms nominal._")

    diag_md = f"""
### 🎯 Diagnostic Conclusion from Stream
**Top Fault:** `{report.top_fault_name}` ({report.confidence_score * 100:.1f}% confidence)  
**Auto-Generated DTCs:** `{', '.join(analysis.extracted_observations.dtc_codes) or 'None'}`  
> **Summary:** {report.executive_summary}
"""
    return metrics_md, anomaly_md, diag_md


def run_post_repair_verification(fault_choice: str, is_repaired: bool):
    """Runs post-repair verification drive cycle and validates monitor completion."""
    fault_type = "healthy" if is_repaired else (
        "vacuum_leak" if "vacuum" in fault_choice else (
            "catalyst_degraded" if "catalyst" in fault_choice else "bms_cell_imbalance"
        )
    )
    stream = generate_telemetry_stream(vehicle_id="generic_petrol", fault_type=fault_type, duration_s=15.0)
    result = verify_post_repair_drive_cycle(fault_id=fault_choice, stream=stream)

    badge = "## 🏁 Verification Outcome: **PASSED (CLEARED FOR DELIVERY)**" if result["verified"] else "## ❌ Verification Outcome: **FAILED (UNRESOLVED)**"
    passed = "\n".join([f"- ✅ {p}" for p in result["passed_checks"]])
    failed = "\n".join([f"- ❌ {f}" for f in result["failed_checks"]])
    monitors = "\n".join([f"- 🟢 {m}" for m in result["monitors_ready"]])

    return f"""
{badge}

### 📋 OBD-II Readiness Monitors
{monitors if monitors else '_No monitors set._'}

### ✅ Passed Operational Checks
{passed if passed else '_None_'}

### ⚠️ Failed Operational Checks
{failed if failed else '_None (Zero Failures)_'}

> **Technician Action:** {result['recommendations']}
"""


with gr.Blocks(title="AI Automotive Digital Twin Diagnostic Suite") as demo:
    gr.Markdown("# 🚗 AI Automotive Digital Twin Diagnostic Suite")
    gr.Markdown("**Supported Subsystems:** Air-Intake & Fuel | Ignition & Combustion | Exhaust & Emissions | EV & Hybrid Electrified Powertrain")

    with gr.Tabs():
        with gr.Tab("🛠️ Multi-Subsystem Diagnostic Station"):
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="Load Benchmark Scenario",
                    choices=list(PRESET_SCENARIOS.keys()),
                    value="-- Select a Preset --",
                )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Technician Intake")
                    vehicle_input = gr.Dropdown(
                        label="Vehicle Platform Profile",
                        choices=list(VEHICLE_OPTIONS.keys()),
                        value="Generic 4-Stroke Petrol Engine (MPI Baseline)",
                    )
                    dtc_input = gr.CheckboxGroup(
                        label="OBD-II Diagnostic Trouble Codes",
                        choices=[
                            "P0101", "P0102", "P0106", "P0107", "P0117", "P0121", "P0128", "P0133", "P0136", "P0171",
                            "P0300", "P0301", "P0335", "P0420", "P0507", "P0A7F", "P0AA6", "P0A93", "P0A1B", "P0A40",
                        ],
                        value=[],
                    )
                    with gr.Accordion("Sensors & Fuel Trims", open=False):
                        idle_q = gr.Dropdown(
                            label="Idle Quality",
                            choices=["Unobserved", "rough", "hunting", "stalling", "normal"],
                            value="Unobserved",
                        )
                        fuel_p = gr.Dropdown(
                            label="Fuel Rail Pressure",
                            choices=["Unobserved", "normal", "low", "high"],
                            value="Unobserved",
                        )
                        fuel_trim = gr.Dropdown(
                            label="Long-Term Fuel Trim (LTFT)",
                            choices=["Unobserved", "high_positive", "high_negative", "fluctuating", "normal"],
                            value="Unobserved",
                        )
                        high_rpm_trim = gr.Dropdown(
                            label="Fuel Trim at 2500 RPM",
                            choices=["Unobserved", "normalizes", "worsens", "unchanged"],
                            value="Unobserved",
                        )
                        maf_read = gr.Dropdown(
                            label="MAF Sensor Reading",
                            choices=["Unobserved", "underreporting", "overreporting", "normal"],
                            value="Unobserved",
                        )
                        hiss = gr.Dropdown(
                            label="Audible Vacuum Hiss",
                            choices=["Unobserved", "present", "absent"],
                            value="Unobserved",
                        )
                    with gr.Accordion("Ignition & Exhaust", open=False):
                        coil_swap = gr.Dropdown(
                            label="Ignition Coil Cylinder Swap Result",
                            choices=["Unobserved", "misfire_follows_coil", "misfire_stays_at_cylinder"],
                            value="Unobserved",
                        )
                        o2_post = gr.Dropdown(
                            label="Downstream O2S2 Voltage Switching Behavior",
                            choices=["Unobserved", "steady_voltage", "active_mirroring_upstream", "fixed_bias"],
                            value="Unobserved",
                        )
                    with gr.Accordion("EV & High-Voltage Powertrain", open=False):
                        cell_delta_in = gr.Dropdown(
                            label="BMS Cell Voltage Delta Under Load",
                            choices=["Unobserved", "normal_balance", "excessive_imbalance"],
                            value="Unobserved",
                        )
                        inv_temp_in = gr.Dropdown(
                            label="Inverter Operating Temperature",
                            choices=["Unobserved", "normal", "abnormally_high"],
                            value="Unobserved",
                        )
                        iso_res_in = gr.Dropdown(
                            label="HV Insulation Resistance to Chassis",
                            choices=["Unobserved", "high_isolation", "loss_of_isolation"],
                            value="Unobserved",
                        )
                    notes_box = gr.Textbox(
                        label="Inspection Notes",
                        placeholder="Technician observations...",
                        lines=2,
                    )
                    shop_rate_input = gr.Number(label="Shop Hourly Labor Rate ($/hr)", value=110.0)
                    run_btn = gr.Button("🚀 Run Digital Twin Diagnosis", variant="primary")

                with gr.Column(scale=1.4):
                    gr.Markdown("### 2. Diagnostic Findings")
                    summary_output = gr.Markdown()
                    hyp_output = gr.DataFrame(label="Ranked Hypotheses (Bayesian Posterior)")
                    question_output = gr.Markdown()
                    with gr.Accordion("🛠️ Automated Work Order & Repair Action Plan", open=True):
                        repair_output = gr.Markdown()
                    with gr.Accordion("📚 Verifiable Technical Citations & Evidence Breakdown", open=False):
                        cit_output = gr.Markdown()
                        ev_output = gr.Markdown()

        with gr.Tab("📡 10Hz Live Telemetry Streamer"):
            gr.Markdown("### Real-Time 10Hz Sensor Telemetry & Anomaly Ingestion")
            with gr.Row():
                stream_vehicle = gr.Dropdown(
                    label="Vehicle Platform",
                    choices=list(VEHICLE_OPTIONS.keys()),
                    value="Volkswagen Golf 1.4L TSI EA211 GDI (2013-2020)",
                )
                stream_fault = gr.Dropdown(
                    label="Drive Cycle Fault Injection",
                    choices=[
                        ("Healthy Baseline (No Fault)", "healthy"),
                        ("Intake Vacuum Leak (Divergent Idle Trims)", "vacuum_leak"),
                        ("Contaminated MAF (Systemic Lean Fueling)", "maf_contamination"),
                        ("Sluggish Upstream O2 Sensor (Hysteresis)", "o2_sensor_sluggish"),
                        ("Degraded Catalytic Converter (Downstream Tracking)", "catalyst_degraded"),
                        ("EV HV Battery Cell Imbalance (Voltage Sag)", "bms_cell_imbalance"),
                        ("EV Inverter Coolant Pump Failure (Thermal Runaway)", "inverter_pump_failure"),
                    ],
                    value="vacuum_leak",
                )
                stream_duration = gr.Slider(minimum=10.0, maximum=30.0, value=20.0, step=5.0, label="Stream Window (Seconds)")
            stream_run_btn = gr.Button("⚡ Ingest & Analyze 10Hz Telemetry Stream", variant="primary")
            
            with gr.Row():
                stream_metrics_out = gr.Markdown()
                stream_anomalies_out = gr.Markdown()
            stream_diag_out = gr.Markdown()

        with gr.Tab("🔄 Post-Repair Verification Drive Cycle"):
            gr.Markdown("### Post-Service Drive Cycle & OBD-II Monitor Reset Engine")
            with gr.Row():
                verify_fault_choice = gr.Dropdown(
                    label="Serviced Failure Mode",
                    choices=[
                        ("Intake Manifold / Vacuum Line Leak", "intake_vacuum_leak"),
                        ("Three-Way Catalytic Converter Low Efficiency", "catalyst_efficiency_degraded"),
                        ("Upstream O2 Sensor Slow Response", "o2_sensor_slow_response"),
                        ("HV Battery Cell Voltage Imbalance", "bms_cell_voltage_imbalance"),
                        ("HV Battery / Inverter Cooling Pump Failure", "battery_cooling_pump_failure"),
                    ],
                    value="intake_vacuum_leak",
                )
                repair_status_choice = gr.Radio(
                    label="Component Physical Status",
                    choices=[("New OEM Part Installed (Repaired)", True), ("Defective Part Left in Place (Unrepaired)", False)],
                    value=True,
                )
            verify_run_btn = gr.Button("🏁 Execute Post-Repair Verification Drive Cycle", variant="primary")
            verify_result_out = gr.Markdown()

    # Wire preset loading
    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[
            vehicle_input, dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim,
            maf_read, hiss, coil_swap, o2_post, cell_delta_in, inv_temp_in, iso_res_in, notes_box,
        ],
    )

    # Wire diagnosis button
    run_btn.click(
        fn=diagnose,
        inputs=[
            vehicle_input, dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim,
            maf_read, hiss, coil_swap, o2_post, cell_delta_in, inv_temp_in, iso_res_in, notes_box, shop_rate_input,
        ],
        outputs=[summary_output, hyp_output, question_output, cit_output, ev_output, repair_output],
    )

    # Wire streaming button
    stream_run_btn.click(
        fn=simulate_and_diagnose_stream,
        inputs=[stream_vehicle, stream_fault, stream_duration],
        outputs=[stream_metrics_out, stream_anomalies_out, stream_diag_out],
    )

    # Wire post-repair verification button
    verify_run_btn.click(
        fn=run_post_repair_verification,
        inputs=[verify_fault_choice, repair_status_choice],
        outputs=[verify_result_out],
    )

if __name__ == "__main__":
    demo.launch()
