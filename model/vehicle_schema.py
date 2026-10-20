"""
Vehicle Profile Schema for Phase 2: Multi-Vehicle Support.
Encapsulates platform-specific powertrain specifications, nominal sensor
ranges, fuel delivery architecture (MPI vs GDI), and known platform failure priors.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NominalSpec(BaseModel):
    min: float
    max: float
    unit: str
    description: Optional[str] = None

    def is_within_spec(self, value: float) -> bool:
        return self.min <= value <= self.max


class VehicleProfile(BaseModel):
    id: str
    name: str
    make: str
    model: str
    engine_code: str
    year_range: str
    fuel_system: str = "MPI"  # "MPI" or "GDI"
    aspiration: str = "Naturally Aspirated"  # "Naturally Aspirated" or "Turbocharged"
    displacement_l: float = 1.2
    nominal_specs: Dict[str, NominalSpec] = Field(default_factory=dict)
    prior_overrides: Dict[str, float] = Field(default_factory=dict)
    additional_faults: List[Dict[str, Any]] = Field(default_factory=list)
    description: Optional[str] = None
