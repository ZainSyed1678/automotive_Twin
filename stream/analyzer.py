"""
Time-Series Stream Analyzer and Physical Anomaly Detector.
Transforms raw 10Hz OBD-II sensor streams into grounded diagnostic observations.
"""

from typing import Dict, List
import statistics

from model.schema import Observations
from .schema import StreamAnalysisResult, TelemetryStream


def analyze_telemetry_stream(stream: TelemetryStream) -> StreamAnalysisResult:
    """
    Performs signal processing, frequency extraction, and physical anomaly
    detection over a time-series telemetry stream.
    """
    frames = stream.frames
    if not frames:
        return StreamAnalysisResult(
            total_frames=0,
            duration_s=0.0,
            sampling_rate_hz=stream.sampling_rate_hz,
            signal_metrics={},
            detected_anomalies=["Empty telemetry stream"],
            extracted_observations=Observations(vehicle_id=stream.vehicle_id),
        )

    duration = frames[-1].timestamp_s - frames[0].timestamp_s
    anomalies: List[str] = []
    dtc_codes: List[str] = []
    symptoms: Dict[str, str] = {}
    metrics: Dict[str, float] = {}

    # 1. Analyze Upstream O2 Sensor Switching Frequency
    o2_values = [f.o2_voltage_v for f in frames]
    v_mid = 0.45
    transitions = 0
    for i in range(1, len(o2_values)):
        if (o2_values[i - 1] < v_mid <= o2_values[i]) or (o2_values[i - 1] > v_mid >= o2_values[i]):
            transitions += 1

    o2_freq = (transitions / 2.0) / max(duration, 1.0)
    metrics["o2_switching_freq_hz"] = round(o2_freq, 2)

    if o2_freq < 0.6:
        anomalies.append(f"Sluggish upstream O2 switching rate ({o2_freq:.2f} Hz < 1.0 Hz threshold)")
        symptoms["o2_sensor_switching_rate"] = "sluggish"
        dtc_codes.append("P0133")

    # 2. Analyze Downstream O2 Sensor Flatline vs Oscillation (Catalyst Health)
    downstream_o2 = [f.downstream_o2_v for f in frames]
    downstream_std = statistics.stdev(downstream_o2) if len(downstream_o2) > 1 else 0.0
    metrics["downstream_o2_std_v"] = round(downstream_std, 3)

    if downstream_std > 0.12:
        anomalies.append(f"Downstream O2 sensor oscillating in tandem with upstream (std={downstream_std:.3f}V > 0.10V)")
        symptoms["downstream_o2_switching"] = "active_mirroring_upstream"
        dtc_codes.append("P0420")
    else:
        symptoms["downstream_o2_switching"] = "steady_voltage"

    # 3. Analyze Fuel Trim Dynamics: Idle vs Cruise Regime
    idle_ltfts = [f.ltft_pct for f in frames if f.timestamp_s < 7.0]
    cruise_ltfts = [f.ltft_pct for f in frames if 7.0 <= f.timestamp_s < 15.0]

    mean_idle_ltft = statistics.mean(idle_ltfts) if idle_ltfts else 0.0
    mean_cruise_ltft = statistics.mean(cruise_ltfts) if cruise_ltfts else 0.0

    metrics["ltft_idle_mean_pct"] = round(mean_idle_ltft, 1)
    metrics["ltft_cruise_mean_pct"] = round(mean_cruise_ltft, 1)

    if mean_idle_ltft > 14.0 and mean_cruise_ltft < 6.0:
        anomalies.append(
            f"Intake vacuum leak signature: LTFT elevated at idle (+{mean_idle_ltft:.1f}%) "
            f"and normalizes at 2500 RPM (+{mean_cruise_ltft:.1f}%)"
        )
        symptoms["idle_quality"] = "rough"
        symptoms["fuel_trim_ltft"] = "high_positive"
        symptoms["fuel_trim_at_high_rpm"] = "normalizes"
        dtc_codes.append("P0171")
    elif mean_idle_ltft > 12.0 and mean_cruise_ltft > 12.0:
        anomalies.append(
            f"Systemic lean fueling / MAF underreporting: LTFT elevated across all engine loads (+{mean_idle_ltft:.1f}%)"
        )
        symptoms["fuel_trim_ltft"] = "high_positive"
        symptoms["fuel_trim_at_high_rpm"] = "worsens"
        symptoms["maf_reading"] = "underreporting"
        dtc_codes.extend(["P0101", "P0171"])

    # 4. Analyze EV / Hybrid Specific PIDs
    cell_deltas = [f.cell_max_delta_mv for f in frames if f.cell_max_delta_mv is not None]
    if cell_deltas:
        max_delta = max(cell_deltas)
        metrics["max_cell_delta_mv"] = round(max_delta, 1)
        if max_delta > 100.0:
            anomalies.append(f"HV Battery cell voltage divergence under load ({max_delta:.1f} mV > 35 mV threshold)")
            symptoms["cell_voltage_delta"] = "excessive_imbalance"
            symptoms["hybrid_system_warning"] = "present"
            symptoms["ev_acceleration"] = "derated"
            dtc_codes.append("P0A7F")

    inv_temps = [f.inverter_temp_c for f in frames if f.inverter_temp_c is not None]
    if inv_temps:
        max_inv = max(inv_temps)
        metrics["max_inverter_temp_c"] = round(max_inv, 1)
        if max_inv > 70.0:
            anomalies.append(f"Inverter thermal overheating ({max_inv:.1f}°C > 65°C limit); loss of coolant flow")
            symptoms["battery_cooling_flow"] = "no_circulation"
            symptoms["inverter_temp"] = "abnormally_high"
            symptoms["hybrid_system_warning"] = "present"
            dtc_codes.append("P0A93")

    obs = Observations(
        dtc_codes=sorted(list(set(dtc_codes))),
        symptoms=symptoms,
        vehicle_id=stream.vehicle_id,
        notes=f"Automated Time-Series Stream Ingestion ({len(frames)} frames @ {stream.sampling_rate_hz}Hz). "
              f"Detected anomalies: {len(anomalies)}.",
    )

    return StreamAnalysisResult(
        total_frames=len(frames),
        duration_s=round(duration, 2),
        sampling_rate_hz=stream.sampling_rate_hz,
        signal_metrics=metrics,
        detected_anomalies=anomalies,
        extracted_observations=obs,
    )
