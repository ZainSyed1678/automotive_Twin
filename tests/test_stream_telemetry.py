"""
Tests for Phase 5: Real-Time Live Sensor Telemetry & Time-Series Streaming Engine.
"""

import pytest
from stream import (
    TelemetryFrame,
    TelemetryStream,
    StreamAnalysisResult,
    generate_telemetry_stream,
    analyze_telemetry_stream,
)
from agents import run_diagnosis


def test_stream_generation_healthy():
    """Verify healthy telemetry generation produces expected frame counts and physics."""
    stream = generate_telemetry_stream(vehicle_id="generic_petrol", fault_type="healthy", duration_s=10.0, sampling_rate_hz=10.0)
    assert len(stream.frames) == 100
    assert stream.frames[0].engine_rpm >= 700.0
    assert stream.frames[0].ltft_pct < 5.0
    assert stream.frames[0].o2_voltage_v >= 0.0


def test_stream_analysis_healthy():
    """Verify healthy stream triggers no severe anomalies or DTCs."""
    stream = generate_telemetry_stream(vehicle_id="generic_petrol", fault_type="healthy", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    assert isinstance(analysis, StreamAnalysisResult)
    assert len(analysis.detected_anomalies) == 0
    assert len(analysis.extracted_observations.dtc_codes) == 0
    assert analysis.signal_metrics["o2_switching_freq_hz"] >= 1.0


def test_vacuum_leak_telemetry_and_diagnosis():
    """Verify vacuum leak idle vs cruise LTFT signature and end-to-end diagnosis."""
    stream = generate_telemetry_stream(vehicle_id="vw_golf_14_tsi", fault_type="vacuum_leak", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("Intake vacuum leak signature" in a for a in analysis.detected_anomalies)
    assert "P0171" in analysis.extracted_observations.dtc_codes
    assert analysis.extracted_observations.symptoms["idle_quality"] == "rough"
    assert analysis.extracted_observations.symptoms["fuel_trim_at_high_rpm"] == "normalizes"
    
    # Run through Bayesian diagnosis pipeline
    diag = run_diagnosis(analysis.extracted_observations)
    assert diag.top_fault_id == "intake_vacuum_leak"
    assert diag.confidence_score >= 0.80


def test_maf_contamination_telemetry():
    """Verify MAF underreporting signature triggers elevated trims and P0101/P0171."""
    stream = generate_telemetry_stream(vehicle_id="generic_petrol", fault_type="maf_contamination", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("Systemic lean fueling" in a for a in analysis.detected_anomalies)
    assert "P0101" in analysis.extracted_observations.dtc_codes
    assert "P0171" in analysis.extracted_observations.dtc_codes
    assert analysis.extracted_observations.symptoms["maf_reading"] == "underreporting"


def test_sluggish_o2_sensor_telemetry_and_diagnosis():
    """Verify O2 switching frequency thresholding and P0133 generation."""
    stream = generate_telemetry_stream(vehicle_id="generic_petrol", fault_type="o2_sensor_sluggish", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("Sluggish upstream O2" in a for a in analysis.detected_anomalies)
    assert "P0133" in analysis.extracted_observations.dtc_codes
    assert analysis.signal_metrics["o2_switching_freq_hz"] < 0.60
    
    diag = run_diagnosis(analysis.extracted_observations)
    assert diag.top_fault_id == "o2_sensor_slow_response"


def test_catalyst_degraded_downstream_mirroring():
    """Verify downstream O2 oscillation detection and P0420 catalytic converter fault."""
    stream = generate_telemetry_stream(vehicle_id="vw_golf_14_tsi", fault_type="catalyst_degraded", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("Downstream O2 sensor oscillating" in a for a in analysis.detected_anomalies)
    assert "P0420" in analysis.extracted_observations.dtc_codes
    assert analysis.signal_metrics["downstream_o2_std_v"] > 0.12
    
    diag = run_diagnosis(analysis.extracted_observations)
    assert diag.top_fault_id == "catalyst_efficiency_degraded"


def test_ev_bms_cell_imbalance_telemetry():
    """Verify high-voltage cell delta under load triggers P0A7F and cell imbalance diagnosis."""
    stream = generate_telemetry_stream(vehicle_id="toyota_prius_18_hybrid", fault_type="bms_cell_imbalance", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("HV Battery cell voltage divergence" in a for a in analysis.detected_anomalies)
    assert "P0A7F" in analysis.extracted_observations.dtc_codes
    assert analysis.signal_metrics["max_cell_delta_mv"] > 100.0
    
    diag = run_diagnosis(analysis.extracted_observations)
    assert diag.top_fault_id == "bms_cell_voltage_imbalance"


def test_ev_inverter_cooling_pump_telemetry():
    """Verify inverter thermal runaway telemetry triggers P0A93 and pump diagnosis."""
    stream = generate_telemetry_stream(vehicle_id="toyota_prius_18_hybrid", fault_type="inverter_pump_failure", duration_s=20.0)
    analysis = analyze_telemetry_stream(stream)
    
    assert any("Inverter thermal overheating" in a for a in analysis.detected_anomalies)
    assert "P0A93" in analysis.extracted_observations.dtc_codes
    assert analysis.signal_metrics["max_inverter_temp_c"] > 70.0
    
    diag = run_diagnosis(analysis.extracted_observations)
    assert diag.top_fault_id == "battery_cooling_pump_failure"


def test_empty_stream_edge_case():
    """Verify analyzer safely handles empty streams without unhandled exceptions."""
    empty_stream = TelemetryStream(vehicle_id="generic_petrol", frames=[])
    analysis = analyze_telemetry_stream(empty_stream)
    assert analysis.total_frames == 0
    assert len(analysis.detected_anomalies) == 1
    assert "Empty telemetry stream" in analysis.detected_anomalies[0]
