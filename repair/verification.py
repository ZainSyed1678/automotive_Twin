"""
Post-Repair Verification Drive Cycle & Monitor Validation Engine.
Analyzes post-repair time-series drive cycle telemetry to verify whether
remedial actions cleared physical fault symptoms and set OBD-II readiness monitors.
"""

from typing import Dict, Any, List
import statistics

from stream.schema import TelemetryStream
from stream.analyzer import analyze_telemetry_stream


def verify_post_repair_drive_cycle(
    fault_id: str,
    stream: TelemetryStream,
) -> Dict[str, Any]:
    """
    Evaluates whether a completed post-repair drive cycle satisfies OEM
    readiness criteria and physical operating parameters for the diagnosed fault.
    """
    analysis = analyze_telemetry_stream(stream)
    frames = stream.frames
    
    passed_checks: List[str] = []
    failed_checks: List[str] = []
    monitors_ready: List[str] = []

    if not frames:
        return {
            "verified": False,
            "readiness_status": "FAILED_NO_TELEMETRY",
            "passed_checks": [],
            "failed_checks": ["No telemetry data received for verification drive cycle."],
            "monitors_ready": [],
            "recommendations": "Conduct the prescribed drive cycle and capture sensor telemetry.",
        }

    # 1. Fault-Specific Physical Checks
    if fault_id == "intake_vacuum_leak":
        idle_ltfts = [f.ltft_pct for f in frames if f.timestamp_s < 7.0]
        mean_idle_ltft = statistics.mean(idle_ltfts) if idle_ltfts else 0.0
        if abs(mean_idle_ltft) <= 6.0:
            passed_checks.append(f"Idle LTFT normalized ({mean_idle_ltft:+.1f}% within [-6%, +6%] spec).")
            monitors_ready.append("Fuel System Monitor: READY")
        else:
            failed_checks.append(f"Idle LTFT remains elevated ({mean_idle_ltft:+.1f}% > 6.0%). Leak persists.")

    elif fault_id == "catalyst_efficiency_degraded":
        downstream_std = analysis.signal_metrics.get("downstream_o2_std_v", 0.0)
        if downstream_std < 0.08:
            passed_checks.append(f"Downstream O2 steady flatline confirmed (std={downstream_std:.3f}V < 0.08V). Catalytic conversion active.")
            monitors_ready.append("Catalyst Monitor: READY")
        else:
            failed_checks.append(f"Downstream O2 still oscillating (std={downstream_std:.3f}V >= 0.08V). Catalyst efficiency insufficient.")

    elif fault_id == "o2_sensor_slow_response":
        o2_freq = analysis.signal_metrics.get("o2_switching_freq_hz", 0.0)
        if o2_freq >= 0.9:
            passed_checks.append(f"O2 sensor switching rate nominal ({o2_freq:.2f} Hz >= 0.9 Hz threshold).")
            monitors_ready.append("Oxygen Sensor Monitor: READY")
        else:
            failed_checks.append(f"O2 sensor switching rate sluggish ({o2_freq:.2f} Hz < 0.9 Hz).")

    elif fault_id == "bms_cell_voltage_imbalance":
        cell_deltas = [f.cell_max_delta_mv for f in frames if f.cell_max_delta_mv is not None]
        max_delta = max(cell_deltas) if cell_deltas else 0.0
        if max_delta <= 35.0:
            passed_checks.append(f"HV Battery cell differential balanced ({max_delta:.1f} mV <= 35 mV under load).")
            monitors_ready.append("HV Battery Management Monitor: READY")
        else:
            failed_checks.append(f"HV Battery cell imbalance remains high ({max_delta:.1f} mV > 35 mV).")

    elif fault_id == "battery_cooling_pump_failure":
        inv_temps = [f.inverter_temp_c for f in frames if f.inverter_temp_c is not None]
        max_temp = max(inv_temps) if inv_temps else 0.0
        if max_temp <= 55.0:
            passed_checks.append(f"Inverter thermal regulation verified ({max_temp:.1f}°C <= 55°C nominal).")
            monitors_ready.append("Inverter Thermal Monitor: READY")
        else:
            failed_checks.append(f"Inverter operating temperature elevated ({max_temp:.1f}°C > 55°C). Check for air pockets.")

    else:
        # Generic check: ensure no DTCs or anomalies
        if not analysis.detected_anomalies:
            passed_checks.append("All engine sensor parameters within factory specifications.")
            monitors_ready.append("Comprehensive Component Monitor: READY")
        else:
            failed_checks.extend(analysis.detected_anomalies)

    # 2. Universal DTC and Anomaly Check
    if len(analysis.extracted_observations.dtc_codes) == 0:
        passed_checks.append("Zero active or pending DTCs logged in ECM/PCM memory.")
    else:
        failed_checks.append(f"Active DTCs still present: {', '.join(analysis.extracted_observations.dtc_codes)}.")

    verified = (len(failed_checks) == 0)

    return {
        "verified": verified,
        "readiness_status": "PASSED_VERIFIED" if verified else "FAILED_UNRESOLVED",
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "monitors_ready": monitors_ready,
        "recommendations": (
            "Repair successfully verified. Vehicle is cleared for customer delivery."
            if verified
            else "Verification failed. Inspect failed criteria, re-smoke intake or inspect wiring before releasing vehicle."
        ),
    }
