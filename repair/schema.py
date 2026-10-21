"""
Data schemas for Phase 6: Automated Repair Action Plan, Labor/Parts Cost Estimator
and Post-Repair Verification Drive Cycle Engine.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PartEstimate(BaseModel):
    """OEM or aftermarket replacement component estimate."""
    part_number: str
    description: str
    quantity: int = 1
    unit_cost_usd: float
    total_cost_usd: float
    category: Literal["oem", "oem_equivalent", "aftermarket"] = "oem"


class LaborEstimate(BaseModel):
    """Standard Mitchell1 / Alldata flat-rate book time and labor calculation."""
    operation: str
    book_hours: float
    hourly_rate_usd: float
    labor_cost_usd: float
    technician_skill: Literal[
        "GENERAL_TECH",
        "ASE_A8_ENGINE_PERFORMANCE",
        "ASE_A6_ELECTRICAL",
        "ASE_L3_HYBRID_EV_SPECIALIST",
    ] = "ASE_A8_ENGINE_PERFORMANCE"


class SafetyWarning(BaseModel):
    """Crucial shop safety warnings (NFPA 70E High Voltage, Fuel System Pressure, Hot Exhaust)."""
    severity: Literal["DANGER_HIGH_VOLTAGE", "WARNING", "CAUTION"]
    standard: str = "OSHA / OEM Standard"
    instruction: str


class RepairStep(BaseModel):
    """Step-by-step Standard Operating Procedure (SOP)."""
    step_number: int
    phase: Literal["PREP_SAFETY", "DISASSEMBLY", "COMPONENT_SERVICE", "REASSEMBLY"]
    instruction: str
    torque_spec: Optional[str] = None
    required_tools: List[str] = Field(default_factory=list)


class DriveCycleStep(BaseModel):
    """Individual dynamic regime in a post-repair verification drive cycle."""
    step_number: int
    phase_name: str
    condition: str
    duration_s: float
    target_speed_kmh: float
    target_rpm: float
    monitor_readiness_check: str


class VerificationProtocol(BaseModel):
    """Standardized post-repair verification protocol and drive cycle routine."""
    monitor_target: str
    drive_cycle_steps: List[DriveCycleStep] = Field(default_factory=list)
    pass_criteria: List[str] = Field(default_factory=list)
    reset_instructions: List[str] = Field(default_factory=list)


class RepairPlan(BaseModel):
    """
    Complete actionable repair plan, cost estimate, safety protocol,
    and post-repair drive cycle verification suite.
    """
    fault_id: str
    fault_name: str
    component_id: str
    vehicle_id: str
    urgency_level: Literal["CRITICAL_IMMEDIATE_STOP", "HIGH_SERVICE_SOON", "MODERATE_SCHEDULE_SERVICE"]
    parts: List[PartEstimate] = Field(default_factory=list)
    labor: List[LaborEstimate] = Field(default_factory=list)
    total_parts_cost_usd: float = 0.0
    total_labor_cost_usd: float = 0.0
    total_estimated_cost_usd: float = 0.0
    safety_warnings: List[SafetyWarning] = Field(default_factory=list)
    procedure_steps: List[RepairStep] = Field(default_factory=list)
    verification_protocol: VerificationProtocol
