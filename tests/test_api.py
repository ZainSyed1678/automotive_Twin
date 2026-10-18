"""
Automated integration tests for the FastAPI Diagnostic Backend.
Uses FastAPI TestClient to verify healthcheck, simulation, search, single-shot diagnosis,
interactive multi-turn diagnosis, and telemetry endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# System & Catalogue Endpoints
# -----------------------------------------------------------------------------

def test_healthcheck(client: TestClient):
    """Verify healthcheck endpoint metadata and catalogue dimensions."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["subsystem"] == "air_intake_and_fuel_control"
    assert data["supported_components_count"] == 10
    assert data["supported_faults_count"] == 12


def test_list_components(client: TestClient):
    """Verify component catalogue retrieval."""
    response = client.get("/model/components")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    comp_ids = [c["id"] for c in data]
    assert "maf_sensor" in comp_ids
    assert "fuel_pump" in comp_ids


def test_list_faults(client: TestClient):
    """Verify fault catalogue retrieval."""
    response = client.get("/model/faults")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 12
    fault_ids = [f["fault_id"] for f in data]
    assert "maf_contamination" in fault_ids
    assert "intake_vacuum_leak" in fault_ids


def test_simulate_fault_valid(client: TestClient):
    """Verify digital twin forward simulation for a valid fault."""
    response = client.post("/simulate/fuel_pump_weak")
    assert response.status_code == 200
    data = response.json()
    assert data["fault_id"] == "fuel_pump_weak"
    assert "P0171" in data["expected_dtcs"]
    assert "fuel_pressure" in data["expected_symptoms"]


def test_simulate_fault_invalid(client: TestClient):
    """Verify 404 response on non-existent fault ID."""
    response = client.post("/simulate/turbocharger_wastegate_stuck")
    assert response.status_code == 404
    assert "not found in catalogue" in response.json()["detail"]


def test_search_knowledge(client: TestClient):
    """Verify hybrid knowledge base search endpoint."""
    response = client.get("/knowledge/search", params={"q": "P0171 fuel lean vacuum leak", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert "citation" in data[0]


# -----------------------------------------------------------------------------
# Diagnostic Reasoning Endpoints
# -----------------------------------------------------------------------------

def test_diagnose_endpoint_straightforward(client: TestClient):
    """Verify single-shot diagnosis endpoint for a straightforward case."""
    payload = {
        "dtc_codes": ["P0101"],
        "symptoms": {
            "maf_reading": "underreporting",
            "acceleration": "hesitation",
        },
        "notes": "Technician reports sluggish acceleration off the line.",
    }
    response = client.post("/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["top_fault_id"] == "maf_contamination"
    assert data["confidence_score"] > 0.70
    assert data["is_ambiguous"] is False
    assert data["review_status"] == "PASSED"
    assert len(data["citations"]) >= 1


def test_diagnose_endpoint_ambiguous(client: TestClient):
    """Verify that ambiguous input triggers an EIG follow-up test question."""
    payload = {
        "dtc_codes": ["P0171"],
        "symptoms": {},
    }
    response = client.post("/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_ambiguous"] is True
    assert data["follow_up_question"] is not None
    q = data["follow_up_question"]
    assert q["test_id"] in ("fuel_pressure", "fuel_trim_at_high_rpm")
    assert q["eig_score"] > 0.10


def test_diagnose_interactive_endpoint(client: TestClient):
    """Verify multi-turn interactive diagnosis endpoint."""
    payload = {
        "dtc_codes": ["P0171"],
        "symptoms": {},
        "answered_tests": {
            "fuel_pressure": "low",
        },
        "max_turns": 3,
    }
    response = client.post("/diagnose/interactive", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["turns_completed"] == 2
    assert len(data["turn_history"]) == 2
    turn1 = data["turn_history"][0]
    turn2 = data["turn_history"][1]

    # Turn 1: ambiguous
    assert turn1["is_ambiguous"] is True
    # Turn 2: resolved towards fuel starvation
    assert turn2["top_fault_id"] == "fuel_pump_weak"
    assert turn2["confidence_score"] > turn1["confidence_score"]


def test_telemetry_traces_endpoint(client: TestClient):
    """Verify that completed diagnostic runs produce telemetry records."""
    response = client.get("/telemetry/traces", params={"limit": 10})
    assert response.status_code == 200
    traces = response.json()
    assert isinstance(traces, list)
    assert len(traces) >= 1
    last_trace = traces[-1]
    assert "trace_id" in last_trace
    assert "duration_ms" in last_trace
    assert "output_metrics" in last_trace
