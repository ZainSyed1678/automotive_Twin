"""
FastAPI REST backend for the AI-Powered Automotive Digital Twin.
Exposes endpoints for single-shot diagnosis, interactive multi-turn disambiguation,
knowledge search, digital twin simulation, and observability traces.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.state import DiagnosticReport
from agents.workflow import run_diagnosis, run_interactive_diagnosis
from model.engine import FaultModel
from model.schema import Observations, SimulationResult
from model.vehicle_registry import vehicle_registry
from rag.store import KnowledgeBase
from stream import (
    TelemetryStream,
    StreamAnalysisResult,
    generate_telemetry_stream,
    analyze_telemetry_stream,
)
from repair import (
    RepairPlan,
    generate_repair_plan,
    verify_post_repair_drive_cycle,
)
from .tracing import get_tracer


app = FastAPI(
    title="Automotive Digital Twin Diagnostic API",
    description="Simulation-Driven Fault Diagnosis API for Vehicle Air-Intake & Fuel-Control Subsystems",
    version="1.0.0",
)

# Enable CORS for local dashboards and interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached model and knowledge base instances
_model: Optional[FaultModel] = None
_kb: Optional[KnowledgeBase] = None


def get_model() -> FaultModel:
    global _model
    if _model is None:
        _model = FaultModel()
    return _model


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


# -----------------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------------

class DiagnoseRequest(BaseModel):
    case_id: Optional[str] = None
    vehicle_id: str = Field(default="generic_petrol", description="Target vehicle platform ID")
    dtc_codes: List[str] = Field(default_factory=list, description="List of OBD-II Diagnostic Trouble Codes")
    symptoms: Dict[str, str] = Field(default_factory=dict, description="Technician observations (e.g. idle_quality=rough)")
    notes: Optional[str] = None


class InteractiveDiagnoseRequest(BaseModel):
    case_id: Optional[str] = None
    vehicle_id: str = Field(default="generic_petrol", description="Target vehicle platform ID")
    dtc_codes: List[str] = Field(default_factory=list)
    symptoms: Dict[str, str] = Field(default_factory=dict)
    answered_tests: Dict[str, str] = Field(
        default_factory=dict,
        description="Follow-up test answers provided by technician (e.g. {'fuel_pressure': 'low'})",
    )
    notes: Optional[str] = None
    max_turns: int = Field(default=3, ge=1, le=5)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def healthcheck() -> Dict[str, Any]:
    """
    Health check returning system status, subsystem boundary, and catalogue dimensions.
    """
    model = get_model()
    subsystems = sorted(list({c.subsystem for c in model.components.values()}))
    return {
        "status": "ok",
        "service": "Automotive Digital Twin Diagnostic Station",
        "subsystem": "air_intake_and_fuel_control",
        "supported_subsystems": subsystems,
        "supported_components_count": len(model.components),
        "supported_faults_count": len(model.faults),
        "supported_vehicles": vehicle_registry.list_vehicle_ids(),
        "version": "1.0.0",
    }


@app.get("/vehicles")
def list_vehicles() -> List[Dict[str, Any]]:
    """
    Returns list of supported vehicle platform profiles with specifications and architecture.
    """
    return [p.model_dump() for p in vehicle_registry.list_profiles()]


@app.get("/vehicles/{vehicle_id}")
def get_vehicle_profile(vehicle_id: str) -> Dict[str, Any]:
    """
    Returns detailed configuration profile for a specific vehicle.
    """
    profile = vehicle_registry.get_profile(vehicle_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle_id}' not found")
    return profile.model_dump()


@app.post("/diagnose", response_model=DiagnosticReport)
def diagnose_case(request: DiagnoseRequest) -> DiagnosticReport:
    """
    Executes the 7-node LangGraph diagnostic pipeline for the given technician observations.
    Returns ranked hypotheses, digital twin evidence, citations, review verification,
    and the optimal EIG follow-up test question if ambiguous.
    """
    obs = Observations(
        dtc_codes=request.dtc_codes,
        symptoms=request.symptoms,
        notes=request.notes,
        vehicle_id=request.vehicle_id,
    )

    start_time = time.perf_counter()
    report = run_diagnosis(obs, case_id=request.case_id)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Record trace
    tracer = get_tracer()
    tracer.record_trace(obs, report, duration_ms, case_id=request.case_id)

    return report


@app.post("/diagnose/interactive")
def interactive_diagnosis(request: InteractiveDiagnoseRequest) -> Dict[str, Any]:
    """
    Executes an interactive multi-turn diagnostic workflow.
    Applies initial observations and sequentially feeds answered tests into the engine,
    returning the diagnostic progression across turns.
    """
    current_obs = Observations(
        dtc_codes=request.dtc_codes,
        symptoms=dict(request.symptoms),
        notes=request.notes,
        vehicle_id=request.vehicle_id,
    )

    reports: List[Dict[str, Any]] = []

    # Turn 1: Initial diagnosis
    turn1_report = run_diagnosis(current_obs, case_id=request.case_id, turn_count=1)
    reports.append(turn1_report.model_dump())

    # Successive turns: apply answered tests
    turn = 2
    for test_id, answer in request.answered_tests.items():
        if turn > request.max_turns:
            break
        current_obs.symptoms[test_id] = str(answer).strip().lower()
        subsequent_report = run_diagnosis(current_obs, case_id=request.case_id, turn_count=turn)
        reports.append(subsequent_report.model_dump())
        turn += 1

    final_report = reports[-1]
    is_resolved = (
        not final_report["is_ambiguous"]
        and final_report["confidence_gap"] >= 0.25
        and not final_report["insufficient_evidence"]
    )

    return {
        "case_id": request.case_id,
        "turns_completed": len(reports),
        "is_resolved": is_resolved,
        "turn_history": reports,
        "final_report": final_report,
    }


@app.get("/model/components")
def list_components() -> List[Dict[str, Any]]:
    """Lists all 10 supported components within the air-intake and fuel-control subsystem."""
    model = get_model()
    return [comp.model_dump() for comp in model.list_components()]


@app.get("/model/faults")
def list_faults() -> List[Dict[str, Any]]:
    """Lists all 12 representative failure modes with their prior distributions."""
    model = get_model()
    return [
        {
            "fault_id": f.fault_id,
            "name": f.name,
            "component_id": f.component_id,
            "prior": f.prior,
            "description": f.description,
            "distinguishing_tests": f.distinguishing_tests,
        }
        for f in model.list_faults()
    ]


@app.post("/simulate/{fault_id}", response_model=SimulationResult)
def simulate_fault(fault_id: str) -> SimulationResult:
    """
    Runs forward causal digital twin simulation for a specific fault ID,
    returning expected DTCs, symptom distributions, and distinguishing tests.
    """
    model = get_model()
    try:
        return model.simulate_fault(fault_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Fault '{fault_id}' not found in catalogue. Call /model/faults for valid IDs.",
        )


@app.get("/knowledge/search")
def search_knowledge(
    q: str = Query(..., description="Technical search query"),
    top_k: int = Query(3, ge=1, le=10),
) -> List[Dict[str, Any]]:
    """Performs hybrid lexical/semantic search across OBD-II code sheets, repair manuals, and TSBs."""
    kb = get_kb()
    results = kb.search(query=q, top_k=top_k)
    return [r.model_dump() for r in results]


@app.get("/telemetry/traces")
def get_telemetry_traces(limit: int = Query(25, ge=1, le=100)) -> List[Dict[str, Any]]:
    """Returns recent structured diagnostic traces with latency and decision audits."""
    tracer = get_tracer()
    return tracer.get_recent_traces(limit=limit)


# -----------------------------------------------------------------------------
# Phase 5: Live Sensor Telemetry & Streaming Endpoints
# -----------------------------------------------------------------------------

class SimulateStreamRequest(BaseModel):
    vehicle_id: str = Field(default="generic_petrol")
    fault_type: str = Field(default="healthy")
    duration_s: float = Field(default=20.0, ge=5.0, le=60.0)
    sampling_rate_hz: float = Field(default=10.0, ge=1.0, le=20.0)


@app.post("/stream/simulate", response_model=TelemetryStream)
def simulate_stream_endpoint(request: SimulateStreamRequest) -> TelemetryStream:
    """
    Simulates a continuous 10Hz physical OBD-II / EV sensor stream with deterministic fault dynamics.
    """
    return generate_telemetry_stream(
        vehicle_id=request.vehicle_id,
        fault_type=request.fault_type,
        duration_s=request.duration_s,
        sampling_rate_hz=request.sampling_rate_hz,
    )


@app.post("/stream/analyze", response_model=StreamAnalysisResult)
def analyze_stream_endpoint(stream: TelemetryStream) -> StreamAnalysisResult:
    """
    Performs time-series signal processing, O2 frequency extraction, LTFT idle/cruise divergence,
    and automatic observation generation from telemetry frames.
    """
    return analyze_telemetry_stream(stream)


@app.post("/stream/diagnose")
def diagnose_from_stream(stream: TelemetryStream) -> Dict[str, Any]:
    """
    Ingests continuous telemetry, extracts physical anomalies and DTCs,
    and executes the full Bayesian digital twin diagnostic engine.
    """
    analysis = analyze_telemetry_stream(stream)
    report = run_diagnosis(analysis.extracted_observations)
    return {
        "analysis": analysis.model_dump(),
        "diagnostic_report": report.model_dump(),
    }


# -----------------------------------------------------------------------------
# Phase 6: Automated Repair Action Plan & Verification Endpoints
# -----------------------------------------------------------------------------

class VerifyRepairRequest(BaseModel):
    fault_id: str
    stream: TelemetryStream


@app.get("/repair/plan/{fault_id}", response_model=RepairPlan)
def get_repair_plan_endpoint(
    fault_id: str,
    vehicle_id: str = Query("generic_petrol", description="Vehicle platform ID"),
    hourly_rate_usd: float = Query(110.0, ge=40.0, le=300.0, description="Shop labor rate per hour"),
) -> RepairPlan:
    """
    Generates an actionable repair plan with OEM part numbers, standard flat-rate book hours,
    NFPA 70E / High Voltage safety protocols, torque specs, and verification drive cycles.
    """
    # Create a synthetic report to route to repair planner
    report = DiagnosticReport(
        top_fault_id=fault_id,
        vehicle_id=vehicle_id,
    )
    return generate_repair_plan(report, hourly_rate_usd=hourly_rate_usd)


@app.post("/repair/verify")
def verify_repair_endpoint(request: VerifyRepairRequest) -> Dict[str, Any]:
    """
    Evaluates post-repair verification drive cycle telemetry to verify whether
    the remedial action successfully cleared fault symptoms and set OBD-II monitors to READY.
    """
    return verify_post_repair_drive_cycle(fault_id=request.fault_id, stream=request.stream)

