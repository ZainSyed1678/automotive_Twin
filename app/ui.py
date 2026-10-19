"""
Streamlit Diagnostic Dashboard for AI-Powered Automotive Digital Twin.
Interactive technician workbench featuring observation intake, benchmark presets,
real-time Bayesian likelihood visualization, EIG follow-up questioning wizard,
and verifiable citation drawer.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when launched via Streamlit CLI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from typing import Dict, List, Optional
import pandas as pd

from agents.workflow import run_diagnosis
from model.engine import FaultModel
from model.schema import Observations
from rag.store import KnowledgeBase
from app.tracing import get_tracer


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Automotive Digital Twin Diagnostic Station",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    .question-box {
        background-color: #EFF6FF;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #BFDBFE;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_fault_model() -> FaultModel:
    return FaultModel()


@st.cache_resource
def load_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


model = load_fault_model()
kb = load_knowledge_base()
tracer = get_tracer()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "dtcs" not in st.session_state:
    st.session_state.dtcs = []
if "symptoms" not in st.session_state:
    st.session_state.symptoms = {}
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "report" not in st.session_state:
    st.session_state.report = None
if "turn_history" not in st.session_state:
    st.session_state.turn_history = []


# -----------------------------------------------------------------------------
# Sidebar: Benchmark Presets & System Metadata
# -----------------------------------------------------------------------------
st.sidebar.title("🚗 Diagnostic Controls")
st.sidebar.markdown("**Subsystem Scope:** Air-Intake & Fuel-Control *(Generic 4-Stroke Petrol)*")

preset_options = [
    "-- Select a Benchmark Scenario --",
    "Scenario 1: MAF Contamination (P0101, Sluggish)",
    "Scenario 2: Ambiguous Lean (P0171 - Follow-up Needed)",
    "Scenario 3: Weak Fuel Pump (P0171, Low Pressure)",
    "Scenario 4: Intake Vacuum Leak (P0171, P0507, Hissing)",
    "Scenario 5: Insufficient Evidence (Empty / Sparse)",
]

selected_preset = st.sidebar.selectbox("Load Benchmark Case", preset_options)

if st.sidebar.button("Load Selected Scenario"):
    if "Scenario 1" in selected_preset:
        st.session_state.dtcs = ["P0101"]
        st.session_state.symptoms = {
            "maf_reading": "underreporting",
            "acceleration": "hesitation",
            "fuel_trim_ltft": "high_positive",
        }
        st.session_state.notes = "Vehicle hesitates on tip-in throttle. MAF sensor hot-wire film suspected."
    elif "Scenario 2" in selected_preset:
        st.session_state.dtcs = ["P0171"]
        st.session_state.symptoms = {}
        st.session_state.notes = "Generic system lean code. Multiple plausible root causes."
    elif "Scenario 3" in selected_preset:
        st.session_state.dtcs = ["P0171"]
        st.session_state.symptoms = {
            "fuel_pressure": "low",
            "fuel_trim_at_high_rpm": "worsens",
        }
        st.session_state.notes = "Fuel rail pressure drops below 2.0 bar under load."
    elif "Scenario 4" in selected_preset:
        st.session_state.dtcs = ["P0171", "P0507"]
        st.session_state.symptoms = {
            "audible_hiss": "present",
            "fuel_trim_at_high_rpm": "normalizes",
            "fuel_pressure": "normal",
        }
        st.session_state.notes = "Audible whistling around intake manifold runner gasket."
    elif "Scenario 5" in selected_preset:
        st.session_state.dtcs = []
        st.session_state.symptoms = {}
        st.session_state.notes = "Customer brought car in with no MIL codes or specific observations."

    st.session_state.report = None
    st.session_state.turn_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f"**Components Modeled:** {len(model.components)}")
st.sidebar.markdown(f"**Failure Modes:** {len(model.faults)}")
st.sidebar.markdown(f"**Technical Docs:** {len(kb.documents)}")
st.sidebar.markdown(f"**Historical Cases Index:** {len(kb.documents)} dev cases")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Automotive Digital Twin Diagnostic Station</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Simulation-Driven Fault Reasoning with Bayesian Likelihoods & Active Questioning</div>',
    unsafe_allow_html=True,
)

tab_workbench, tab_model, tab_rag, tab_telemetry = st.tabs(
    ["Diagnostic Workbench", "Digital Twin Model", "Technical Documents (RAG)", "Telemetry & Traces"]
)


# -----------------------------------------------------------------------------
# TAB 1: Diagnostic Workbench
# -----------------------------------------------------------------------------
with tab_workbench:
    col_intake, col_output = st.columns([1, 1.3], gap="medium")

    with col_intake:
        st.subheader("1. Technician Intake")

        # Available DTC catalogue
        all_known_dtcs = [
            "P0101", "P0102", "P0106", "P0107", "P0117", "P0121", "P0128",
            "P0133", "P0171", "P0300", "P0301", "P0507",
        ]

        selected_dtcs = st.multiselect(
            "Diagnostic Trouble Codes (OBD-II)",
            options=all_known_dtcs,
            default=st.session_state.dtcs,
            help="Select all active fault codes scanned from the ECU.",
        )

        st.markdown("**Physical Observations & Sensor Telemetry:**")

        c1, c2 = st.columns(2)
        with c1:
            idle_q = st.selectbox(
                "Idle Quality",
                ["Unobserved", "rough", "hunting", "stalling", "normal"],
                index=["Unobserved", "rough", "hunting", "stalling", "normal"].index(
                    st.session_state.symptoms.get("idle_quality", "Unobserved")
                ),
            )
            fuel_p = st.selectbox(
                "Fuel Rail Pressure",
                ["Unobserved", "normal", "low", "high"],
                index=["Unobserved", "normal", "low", "high"].index(
                    st.session_state.symptoms.get("fuel_pressure", "Unobserved")
                ),
            )
            fuel_trim = st.selectbox(
                "Long-Term Fuel Trim (LTFT)",
                ["Unobserved", "high_positive", "high_negative", "fluctuating", "normal"],
                index=["Unobserved", "high_positive", "high_negative", "fluctuating", "normal"].index(
                    st.session_state.symptoms.get("fuel_trim_ltft", "Unobserved")
                ),
            )

        with c2:
            maf_read = st.selectbox(
                "MAF Sensor Reading",
                ["Unobserved", "underreporting", "overreporting", "normal"],
                index=["Unobserved", "underreporting", "overreporting", "normal"].index(
                    st.session_state.symptoms.get("maf_reading", "Unobserved")
                ),
            )
            high_rpm_trim = st.selectbox(
                "Fuel Trim at 2500 RPM",
                ["Unobserved", "normalizes", "worsens", "unchanged"],
                index=["Unobserved", "normalizes", "worsens", "unchanged"].index(
                    st.session_state.symptoms.get("fuel_trim_at_high_rpm", "Unobserved")
                ),
            )
            hiss = st.selectbox(
                "Audible Vacuum Hiss",
                ["Unobserved", "present", "absent"],
                index=["Unobserved", "present", "absent"].index(
                    st.session_state.symptoms.get("audible_hiss", "Unobserved")
                ),
            )

        notes_input = st.text_area(
            "Technician Inspection Notes",
            value=st.session_state.notes,
            placeholder="e.g. Engine runs unevenly at cold idle, customer noticed whistling sound...",
            height=70,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_btn = st.button("🚀 Run Digital Twin Diagnosis", type="primary", use_container_width=True)
        with btn_col2:
            clear_btn = st.button("🔄 Reset Observations", use_container_width=True)

        if clear_btn:
            st.session_state.dtcs = []
            st.session_state.symptoms = {}
            st.session_state.notes = ""
            st.session_state.report = None
            st.session_state.turn_history = []
            st.rerun()

        if run_btn:
            # Compile symptoms
            symptom_dict = {}
            if idle_q != "Unobserved":
                symptom_dict["idle_quality"] = idle_q
            if fuel_p != "Unobserved":
                symptom_dict["fuel_pressure"] = fuel_p
            if fuel_trim != "Unobserved":
                symptom_dict["fuel_trim_ltft"] = fuel_trim
            if maf_read != "Unobserved":
                symptom_dict["maf_reading"] = maf_read
            if high_rpm_trim != "Unobserved":
                symptom_dict["fuel_trim_at_high_rpm"] = high_rpm_trim
            if hiss != "Unobserved":
                symptom_dict["audible_hiss"] = hiss

            st.session_state.dtcs = selected_dtcs
            st.session_state.symptoms = symptom_dict
            st.session_state.notes = notes_input

            obs = Observations(
                dtc_codes=selected_dtcs,
                symptoms=symptom_dict,
                notes=notes_input,
            )

            with st.spinner("Executing 7-node LangGraph digital twin workflow..."):
                report = run_diagnosis(obs)
                st.session_state.report = report
                st.session_state.turn_history = [report]
                st.rerun()

    with col_output:
        st.subheader("2. Digital Twin Diagnostic Findings")

        report = st.session_state.report

        if not report:
            st.info(
                "👈 Enter diagnostic trouble codes and symptoms on the left, "
                "or load a preset scenario from the sidebar and click **Run Digital Twin Diagnosis**."
            )
        else:
            # 1. Review Status Badge & Overconfidence Guard
            if report.review_status == "PASSED":
                st.success("🛡️ **Self-Review Gate: PASSED** — Evidence strictly grounded; scope verified.")
            else:
                st.warning(f"⚠️ **Self-Review Gate: {report.review_status}**")

            # 2. Key Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    label="Primary Hypothesis",
                    value=report.top_fault_name or "Indeterminate",
                    delta=f"{report.confidence_score * 100:.1f}% Posterior",
                )
            with m2:
                gap_val = report.confidence_gap * 100
                st.metric(
                    label="Confidence Margin",
                    value=f"{gap_val:.1f}%",
                    delta="Ambiguous" if report.is_ambiguous else "Distinctive",
                    delta_color="off" if report.is_ambiguous else "normal",
                )
            with m3:
                status_label = "Insufficient Evidence" if report.insufficient_evidence else (
                    "Ambiguous" if report.is_ambiguous else "High Confidence"
                )
                st.metric(label="Diagnostic Status", value=status_label)

            # 3. Executive Summary
            if report.insufficient_evidence:
                st.warning(f"**Notice:** {report.executive_summary}")
            elif report.is_ambiguous:
                st.info(f"**Analysis:** {report.executive_summary}")
            else:
                st.success(f"**Conclusion:** {report.executive_summary}")

            # 4. Ranked Hypotheses Distribution Chart
            st.markdown("#### Hypothesis Posterior Probability Distribution")
            chart_data = []
            for h in report.ranked_hypotheses:
                chart_data.append({
                    "Fault": h["fault_name"],
                    "Probability": round(h["posterior_prob"] * 100.0, 1),
                })
            df = pd.DataFrame(chart_data)
            st.bar_chart(df.set_index("Fault"), color="#2563EB", horizontal=True)

            # 5. Adaptive Follow-up Question Wizard (EIG Active Questioning)
            follow_up = report.follow_up_question
            if follow_up:
                st.markdown(
                    f"""
                    <div class="question-box">
                        <h4 style="margin-top:0; color:#1E3A8A;">❓ Targeted Follow-up Inspection (EIG: {follow_up['eig_score']:.3f} bits)</h4>
                        <p style="font-size:1.05rem; font-weight:600; margin-bottom:4px;">{follow_up['question_text']}</p>
                        <p style="color:#4B5563; font-size:0.9rem; margin-bottom:10px;"><em>Rationale: {follow_up['rationale']}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Select test result to update diagnosis dynamically:**")
                btn_cols = st.columns(len(follow_up["possible_answers"]))
                for idx, ans in enumerate(follow_up["possible_answers"]):
                    with btn_cols[idx]:
                        if st.button(f"Result: {ans.upper()}", key=f"ans_btn_{ans}", use_container_width=True):
                            # Inject test result into symptoms and re-run
                            st.session_state.symptoms[follow_up["test_id"]] = ans
                            new_obs = Observations(
                                dtc_codes=st.session_state.dtcs,
                                symptoms=st.session_state.symptoms,
                                notes=st.session_state.notes,
                            )
                            updated_report = run_diagnosis(new_obs)
                            st.session_state.report = updated_report
                            st.session_state.turn_history.append(updated_report)
                            st.rerun()

            # 6. Verifiable Citations Drawer
            with st.expander(f"📚 Verifiable Technical Citations ({len(report.citations)} Attached)"):
                for cit in report.citations:
                    st.markdown(f"- `{cit}`")

            # 7. Grounded Evidence Breakdown
            with st.expander("🔍 Evidence Matches & Physical Contradictions"):
                c_match, c_contra = st.columns(2)
                with c_match:
                    st.markdown("**Evidence Matches:**")
                    if report.supporting_evidence:
                        for m in report.supporting_evidence:
                            st.markdown(f"- ✅ `{m}`")
                    else:
                        st.caption("No specific evidence matches identified.")
                with c_contra:
                    st.markdown("**Physical Contradictions:**")
                    if report.contradictory_evidence:
                        for c in report.contradictory_evidence:
                            st.markdown(f"- ❌ `{c}`")
                    else:
                        st.caption("No physical contradictions detected.")


# -----------------------------------------------------------------------------
# TAB 2: Digital Twin Causal Explorer
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("Subsystem Causal Architecture & Failure Modes")
    st.markdown(
        "Explore the 10 components and 12 representative failure modes modeled in the "
        "air-intake and fuel-control digital twin."
    )

    col_comps, col_sim = st.columns([1, 1.2], gap="large")

    with col_comps:
        st.markdown("### 10 Subsystem Components")
        for comp in model.list_components():
            with st.container():
                st.markdown(f"**{comp.name}** (`{comp.id}`)")
                st.caption(comp.description)
                st.divider()

    with col_sim:
        st.markdown("### Forward Fault Simulation Sandbox")
        sim_fault_id = st.selectbox(
            "Select Failure Mode to Simulate",
            options=[f.fault_id for f in model.list_faults()],
            format_func=lambda fid: model.get_fault(fid).name if model.get_fault(fid) else fid,
        )

        if st.button("Run Forward Simulation", key="btn_run_sim"):
            sim_res = model.simulate_fault(sim_fault_id)
            st.markdown(f"#### Simulation Results for `{sim_res.fault_name}`")
            st.write(sim_res.description)

            st.markdown("**Expected Diagnostic Trouble Codes:**")
            st.json(sim_res.expected_dtcs)

            st.markdown("**Expected Symptom Likelihoods:**")
            st.json(sim_res.expected_symptoms)

            st.markdown("**Recommended Distinguishing Tests:**")
            for t in sim_res.distinguishing_tests:
                st.markdown(f"- 🔬 {t}")


# -----------------------------------------------------------------------------
# TAB 3: Technical Documents (RAG)
# -----------------------------------------------------------------------------
with tab_rag:
    st.subheader("Technical Repair Manuals, OBD Sheets & TSBs")
    search_q = st.text_input(
        "Search Knowledge Base",
        value="P0171 lean fuel vacuum leak",
        placeholder="Enter DTC code, symptom, or repair keyword...",
    )
    if st.button("Search Knowledge Base"):
        hits = kb.search(query=search_q, top_k=5)
        st.markdown(f"**Found {len(hits)} matching technical documents:**")
        for hit in hits:
            st.markdown(f"### {hit.title} (`{hit.citation}`)")
            st.caption(f"Category: {hit.doc_type.upper()} | Score: {hit.score:.3f}")
            st.text(hit.content)
            st.divider()


# -----------------------------------------------------------------------------
# TAB 4: Telemetry & Traces
# -----------------------------------------------------------------------------
with tab_telemetry:
    st.subheader("Observability & Telemetry Audit Log")
    traces = tracer.get_recent_traces(limit=20)
    if not traces:
        st.caption("No diagnostic traces logged in current session.")
    else:
        st.dataframe(
            pd.DataFrame([
                {
                    "Trace ID": t["trace_id"][:8],
                    "Timestamp": t["timestamp"],
                    "Duration (ms)": t["duration_ms"],
                    "DTCs": ", ".join(t["input_metrics"]["dtc_codes"]),
                    "Top Fault": t["output_metrics"]["top_fault_id"],
                    "Confidence": t["output_metrics"]["confidence_score"],
                    "Margin": t["output_metrics"]["confidence_gap"],
                    "Review Status": t["output_metrics"]["review_status"],
                    "EIG Question": t["adaptive_questioning"]["test_id"],
                }
                for t in reversed(traces)
            ]),
            use_container_width=True,
        )
