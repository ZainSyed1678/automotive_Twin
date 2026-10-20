"""
Automotive Digital Twin Diagnostic Station - Gradio Interface.
Deployment-ready entrypoint for Hugging Face Spaces (Gradio SDK).
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


VEHICLE_OPTIONS = {
    "Generic 4-Stroke Petrol Engine (MPI Baseline)": "generic_petrol",
    "Hyundai i20 1.2L Kappa II MPI (2014-2020)": "hyundai_i20_kappa_12",
    "Volkswagen Golf 1.4L TSI EA211 GDI (2013-2020)": "vw_golf_14_tsi",
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
        "idle_quality": "Unobserved",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "normalizes",
        "maf_reading": "Unobserved",
        "audible_hiss": "present",
        "notes": "Audible whistling around intake manifold runner gasket.",
    },
    "Scenario 5: Insufficient Evidence (Empty / Sparse)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": [],
        "idle_quality": "Unobserved",
        "fuel_pressure": "Unobserved",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "notes": "Customer brought car in with no MIL codes or specific observations.",
    },
    "Scenario 6: Hyundai i20 Purge Valve Sticking (P0171, P0507, Rough Idle)": {
        "vehicle": "Hyundai i20 1.2L Kappa II MPI (2014-2020)",
        "dtcs": ["P0171", "P0507"],
        "idle_quality": "rough",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "high_positive",
        "fuel_trim_at_high_rpm": "normalizes",
        "maf_reading": "Unobserved",
        "audible_hiss": "present",
        "notes": "Hyundai Kappa 1.2L rough idle especially after refueling. EVAP purge control valve suspected.",
    },
    "Scenario 7: VW Golf EA211 Direct Injection Carbon Fouling (P0300, Rough Cold Start)": {
        "vehicle": "Volkswagen Golf 1.4L TSI EA211 GDI (2013-2020)",
        "dtcs": ["P0300"],
        "idle_quality": "rough",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "high_positive",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "ignition_coil_swap": "Unobserved",
        "downstream_o2": "Unobserved",
        "notes": "VW 1.4 TSI EA211 rough idle on cold start. Direct injection intake valve carbon buildup suspected.",
    },
    "Scenario 8: Ignition Coil Breakdown with Catalyst Risk (P0301, Misfire Under Load)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0300", "P0301"],
        "idle_quality": "rough",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "ignition_coil_swap": "misfire_follows_coil",
        "downstream_o2": "Unobserved",
        "notes": "Engine hesitates sharply under acceleration. Misfire moves with swapped coil pack. High risk of catalyst meltdown.",
    },
    "Scenario 9: Catalytic Converter Degradation (P0420, Downstream O2 Active)": {
        "vehicle": "Generic 4-Stroke Petrol Engine (MPI Baseline)",
        "dtcs": ["P0420"],
        "idle_quality": "normal",
        "fuel_pressure": "normal",
        "fuel_trim_ltft": "normal",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "ignition_coil_swap": "Unobserved",
        "downstream_o2": "active_mirroring_upstream",
        "notes": "Catalyst efficiency DTC stored. Post-catalytic O2 sensor oscillates in tandem with upstream sensor.",
    },
}


def load_preset(preset_name: str):
    data = PRESET_SCENARIOS.get(preset_name, PRESET_SCENARIOS["-- Select a Preset --"])
    return (
        data.get("vehicle", "Generic 4-Stroke Petrol Engine (MPI Baseline)"),
        data["dtcs"],
        data["idle_quality"],
        data["fuel_pressure"],
        data["fuel_trim_ltft"],
        data["fuel_trim_at_high_rpm"],
        data["maf_reading"],
        data["audible_hiss"],
        data.get("ignition_coil_swap", "Unobserved"),
        data.get("downstream_o2", "Unobserved"),
        data["notes"],
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
    notes: str,
):
    vehicle_id = VEHICLE_OPTIONS.get(vehicle_name, "generic_petrol")

    # Assemble symptom dict
    symptoms = {}
    if idle_quality and idle_quality != "Unobserved":
        symptoms["idle_quality"] = idle_quality
    if fuel_pressure and fuel_pressure != "Unobserved":
        symptoms["fuel_pressure"] = fuel_pressure
    if fuel_trim_ltft and fuel_trim_ltft != "Unobserved":
        symptoms["fuel_trim_ltft"] = fuel_trim_ltft
    if fuel_trim_at_high_rpm and fuel_trim_at_high_rpm != "Unobserved":
        symptoms["fuel_trim_at_high_rpm"] = fuel_trim_at_high_rpm
    if maf_reading and maf_reading != "Unobserved":
        symptoms["maf_reading"] = maf_reading
    if audible_hiss and audible_hiss != "Unobserved":
        symptoms["audible_hiss"] = audible_hiss
    if ignition_coil_swap and ignition_coil_swap != "Unobserved":
        symptoms["ignition_coil_swap"] = ignition_coil_swap
    if downstream_o2 and downstream_o2 != "Unobserved":
        symptoms["downstream_o2_switching"] = downstream_o2

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

    return summary_md, hyp_df, q_md, cit_md, ev_md


with gr.Blocks(title="Automotive Multi-Subsystem Digital Twin") as demo:
    gr.Markdown("# 🚗 Automotive Multi-Subsystem Digital Twin Diagnostic Station")
    gr.Markdown("**Subsystems:** Air-Intake & Fuel-Control | Ignition & Combustion | Exhaust & Aftertreatment")

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
                label="Vehicle Platform Profile (Phase 2 Multi-Vehicle)",
                choices=list(VEHICLE_OPTIONS.keys()),
                value="Generic 4-Stroke Petrol Engine (MPI Baseline)",
            )
            dtc_input = gr.CheckboxGroup(
                label="OBD-II Diagnostic Trouble Codes",
                choices=["P0101", "P0102", "P0106", "P0107", "P0117", "P0121", "P0128", "P0133", "P0136", "P0171", "P0300", "P0301", "P0335", "P0420", "P0507"],
                value=[],
            )
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
            notes_box = gr.Textbox(
                label="Inspection Notes",
                placeholder="Technician observations...",
                lines=2,
            )
            run_btn = gr.Button("🚀 Run Digital Twin Diagnosis", variant="primary")

        with gr.Column(scale=1.4):
            gr.Markdown("### 2. Diagnostic Findings")
            summary_output = gr.Markdown()
            hyp_output = gr.DataFrame(label="Ranked Hypotheses (Bayesian Posterior)")
            question_output = gr.Markdown()
            with gr.Accordion("📚 Verifiable Technical Citations & Evidence Breakdown", open=False):
                cit_output = gr.Markdown()
                ev_output = gr.Markdown()

    # Wire preset loading
    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[vehicle_input, dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim, maf_read, hiss, coil_swap, o2_post, notes_box],
    )

    # Wire diagnosis button
    run_btn.click(
        fn=diagnose,
        inputs=[vehicle_input, dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim, maf_read, hiss, coil_swap, o2_post, notes_box],
        outputs=[summary_output, hyp_output, question_output, cit_output, ev_output],
    )

if __name__ == "__main__":
    demo.launch()
