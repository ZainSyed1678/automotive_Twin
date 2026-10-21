"""
Real-Time Telemetry Generator.
Simulates realistic 10Hz physical OBD-II sensor streams across standardized
drive cycles, with deterministic physical fault injection.
"""

import math
from typing import Optional

from .schema import TelemetryFrame, TelemetryStream


def generate_telemetry_stream(
    vehicle_id: str = "generic_petrol",
    fault_type: str = "healthy",
    duration_s: float = 20.0,
    sampling_rate_hz: float = 10.0,
) -> TelemetryStream:
    """
    Simulates a continuous 10Hz OBD-II telemetry stream across an urban drive cycle:
    - 0s to 7s: Warm Curb Idle (750 RPM, closed throttle)
    - 7s to 15s: Steady Cruise (2500 RPM, 60 km/h)
    - 15s to 20s: Moderate Acceleration / Deceleration
    """
    total_steps = int(duration_s * sampling_rate_hz)
    dt = 1.0 / sampling_rate_hz
    frames = []

    is_ev = "bev" in vehicle_id or "hybrid" in vehicle_id

    for i in range(total_steps):
        t = i * dt

        # Determine drive cycle regime
        if t < 7.0:
            rpm = 750.0 + 15.0 * math.sin(2.0 * math.pi * 0.8 * t)
            speed = 0.0
            throttle = 0.0
            map_val = 32.0 + 1.5 * math.sin(t)
            maf_nominal = 2.2
        elif t < 15.0:
            rpm = 2500.0 + 25.0 * math.sin(2.0 * math.pi * 0.5 * t)
            speed = 60.0
            throttle = 22.0
            map_val = 55.0 + 2.0 * math.sin(t)
            maf_nominal = 12.5
        else:
            rpm = 1800.0 + 40.0 * math.sin(t)
            speed = 45.0
            throttle = 15.0
            map_val = 45.0
            maf_nominal = 8.0

        # Normal closed loop upstream O2 oscillation (nominal ~ 1.5 Hz)
        o2_freq = 1.6
        if fault_type == "o2_sensor_sluggish":
            o2_freq = 0.25  # Severe response hysteresis

        o2_voltage = 0.48 + 0.38 * math.sin(2.0 * math.pi * o2_freq * t)
        o2_voltage = max(0.08, min(0.92, o2_voltage))

        # Downstream O2 sensor behavior
        if fault_type == "catalyst_degraded":
            # Mirrors upstream O2 oscillation
            downstream_o2 = 0.48 + 0.32 * math.sin(2.0 * math.pi * o2_freq * t - 0.4)
        else:
            # Normal: steady flatline between 0.60V and 0.68V
            downstream_o2 = 0.64 + 0.02 * math.sin(0.2 * t)
        downstream_o2 = max(0.08, min(0.92, downstream_o2))

        # Fuel Trim Physics
        if fault_type == "vacuum_leak":
            # High trim at idle (+21%), normalizes at 2500 RPM (+2.5%)
            if t < 7.0:
                ltft = 21.0 + 1.2 * math.sin(t)
                stft = 4.0 * math.sin(3.0 * t)
            elif t < 15.0:
                ltft = 2.5 + 0.8 * math.sin(t)
                stft = 1.5 * math.sin(2.0 * t)
            else:
                ltft = 6.0
                stft = 1.0
            maf_actual = maf_nominal
        elif fault_type == "maf_contamination":
            # MAF underreports by 30%; trims remain elevated across whole RPM range (+18%)
            maf_actual = maf_nominal * 0.70
            ltft = 18.5 + 1.0 * math.sin(t)
            stft = 3.0 * math.sin(2.0 * t)
        else:
            maf_actual = maf_nominal
            ltft = 1.5 + 0.8 * math.sin(t)
            stft = 1.0 * math.sin(3.0 * t)

        # EV / Hybrid PIDs
        hv_v = None
        hv_a = None
        cell_delta = None
        inv_temp = None

        if is_ev:
            nominal_pack = 380.0 if "bev" in vehicle_id else 210.0
            current_draw = 15.0 if t < 7.0 else (85.0 if t < 15.0 else 30.0)
            hv_a = current_draw
            hv_v = nominal_pack - current_draw * 0.12

            if fault_type == "bms_cell_imbalance":
                # Cell delta spikes under high current draw
                cell_delta = 135.0 if current_draw > 50.0 else 45.0
            else:
                cell_delta = 18.0 + 4.0 * math.sin(t)

            if fault_type == "inverter_pump_failure":
                # Inverter temperature overheats steadily
                inv_temp = 42.0 + (t * 2.1)
            else:
                inv_temp = 44.0 + 2.0 * math.sin(0.1 * t)

        frame = TelemetryFrame(
            timestamp_s=round(t, 2),
            engine_rpm=round(rpm, 1),
            vehicle_speed_kmh=round(speed, 1),
            throttle_pos_pct=round(throttle, 1),
            maf_gps=round(maf_actual, 2),
            map_kpa=round(map_val, 1),
            coolant_temp_c=88.0,
            stft_pct=round(stft, 1),
            ltft_pct=round(ltft, 1),
            o2_voltage_v=round(o2_voltage, 3),
            downstream_o2_v=round(downstream_o2, 3),
            hv_battery_voltage_v=round(hv_v, 1) if hv_v is not None else None,
            hv_battery_current_a=round(hv_a, 1) if hv_a is not None else None,
            cell_max_delta_mv=round(cell_delta, 1) if cell_delta is not None else None,
            inverter_temp_c=round(inv_temp, 1) if inv_temp is not None else None,
        )
        frames.append(frame)

    return TelemetryStream(
        vehicle_id=vehicle_id,
        simulated_fault=fault_type,
        sampling_rate_hz=sampling_rate_hz,
        frames=frames,
    )
