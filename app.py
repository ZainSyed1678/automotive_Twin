"""
Automotive Digital Twin Diagnostic Station - Gradio Interface.
Deployment-ready entrypoint for Hugging Face Spaces (Gradio SDK).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from agents.workflow import run_diagnosis
from model.schema import Observations


PRESET_SCENARIOS = {
    "-- Select a Preset --": {
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
        "dtcs": [],
        "idle_quality": "Unobserved",
        "fuel_pressure": "Unobserved",
        "fuel_trim_ltft": "Unobserved",
        "fuel_trim_at_high_rpm": "Unobserved",
        "maf_reading": "Unobserved",
        "audible_hiss": "Unobserved",
        "notes": "Customer brought car in with no MIL codes or specific observations.",
    },
}


def load_preset(preset_name: str):
    data = PRESET_SCENARIOS.get(preset_name, PRESET_SCENARIOS["-- Select a Preset --"])
    return (
        data["dtcs"],
        data["idle_quality"],
        data["fuel_pressure"],
        data["fuel_trim_ltft"],
        data["fuel_trim_at_high_rpm"],
        data["maf_reading"],
        data["audible_hiss"],
        data["notes"],
    )


def diagnose(
    dtcs: List[str],
    idle_quality: str,
    fuel_pressure: str,
    fuel_trim_ltft: str,
    fuel_trim_at_high_rpm: str,
    maf_reading: str,
    audible_hiss: str,
    notes: str,
):
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

    obs = Observations(
        dtc_codes=dtcs or [],
        symptoms=symptoms,
        notes=notes or None,
    )

    report = run_diagnosis(obs)

    # 1. Status banner
    if report.review_status == "PASSED":
        badge = "### 🛡️ **Self-Review Gate: PASSED** (Grounding & Subsystem Scope Verified)"
    else:
        badge = f"### ⚠️ **Self-Review Gate: {report.review_status}**"

    # 2. Executive summary
    summary_md = f"""
{badge}

**Top Hypothesis:** `{report.top_fault_name or 'Indeterminate'}`  
**Posterior Confidence:** `{report.confidence_score * 100:.1f}%` | **Margin Gap:** `{report.confidence_gap * 100:.1f}%`  
**Diagnostic Status:** `{'Insufficient Evidence' if report.insufficient_evidence else ('Ambiguous' if report.is_ambiguous else 'High Confidence')}`

> **Summary:** {report.executive_summary}
"""

    # 3. Ranked hypotheses table
    table_rows = []
    for idx, h in enumerate(report.ranked_hypotheses[:5], 1):
        table_rows.append({
            "Rank": idx,
            "Failure Mode": h.get("fault_name", ""),
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


with gr.Blocks(title="Automotive Digital Twin Diagnostic Station") as demo:
    gr.Markdown("# 🚗 Automotive Digital Twin Diagnostic Station")
    gr.Markdown("**Subsystem:** Air-Intake & Fuel-Control *(Generic 4-Stroke Petrol Engine)*")

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            label="Load Benchmark Scenario",
            choices=list(PRESET_SCENARIOS.keys()),
            value="-- Select a Preset --",
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Technician Intake")
            dtc_input = gr.CheckboxGroup(
                label="OBD-II Diagnostic Trouble Codes",
                choices=["P0101", "P0102", "P0106", "P0107", "P0117", "P0121", "P0128", "P0133", "P0171", "P0300", "P0301", "P0507"],
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
        outputs=[dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim, maf_read, hiss, notes_box],
    )

    # Wire diagnosis button
    run_btn.click(
        fn=diagnose,
        inputs=[dtc_input, idle_q, fuel_p, fuel_trim, high_rpm_trim, maf_read, hiss, notes_box],
        outputs=[summary_output, hyp_output, question_output, cit_output, ev_output],
    )

if __name__ == "__main__":
    demo.launch()
