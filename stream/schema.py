"""
Schemas for Real-Time OBD-II Live Telemetry and Time-Series Streaming Engine.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from model.schema import Observations


class TelemetryFrame(BaseModel):
    """
    A single temporal snapshot of physical powertrain sensor telemetry.
    """
    timestamp_s: float
    engine_rpm: float = 800.0
    vehicle_speed_kmh: float = 0.0
    throttle_pos_pct: float = 0.0
    maf_gps: float = 2.4
    map_kpa: float = 35.0
    coolant_temp_c: float = 88.0
    stft_pct: float = 0.0
    ltft_pct: float = 0.0
    o2_voltage_v: float = 0.45
    downstream_o2_v: float = 0.65

    # EV / Hybrid specific PIDs
    hv_battery_voltage_v: Optional[float] = None
    hv_battery_current_a: Optional[float] = None
    cell_max_delta_mv: Optional[float] = None
    inverter_temp_c: Optional[float] = None


class TelemetryStream(BaseModel):
    """
    Continuous series of telemetry frames over time window [0, T].
    """
    vehicle_id: str = "generic_petrol"
    simulated_fault: Optional[str] = None
    sampling_rate_hz: float = 10.0
    frames: List[TelemetryFrame] = Field(default_factory=list)


class StreamAnalysisResult(BaseModel):
    """
    Outcome of time-series signal processing, frequency extraction,
    and automatic anomaly detection.
    """
    total_frames: int
    duration_s: float
    sampling_rate_hz: float
    signal_metrics: Dict[str, float]
    detected_anomalies: List[str]
    extracted_observations: Observations
