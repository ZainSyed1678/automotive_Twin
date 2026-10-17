"""
Schemas for Ground-Truth and Agent-Visible Diagnostic Cases.
Enforces strict boundary separation between evaluation data and agent input.
"""

from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict, Field

from model.schema import Observations


class DifficultyLevel(str, Enum):
    STRAIGHTFORWARD = "straightforward"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    MISLEADING = "misleading"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AgentCase(BaseModel):
    """
    Sanitized, agent-visible diagnostic case.
    Strictly forbids any extra attributes to guarantee zero ground-truth leakage.
    """
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., description="Unique case identifier")
    observations: Observations = Field(..., description="Observed DTCs and symptoms")


class GroundTruthCase(BaseModel):
    """
    Internal evaluation case containing the ground-truth fault and generation details.
    This schema is NEVER provided to the diagnostic agent or MCP tools.
    """
    case_id: str
    true_fault_id: str
    difficulty: DifficultyLevel
    observations: Observations
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_agent_case(self) -> AgentCase:
        """Projects this ground-truth case into an immutable, sanitized AgentCase."""
        return AgentCase(
            case_id=self.case_id,
            observations=self.observations.model_copy(deep=True)
        )

    def to_agent_dict(self) -> Dict[str, Any]:
        """
        Serializes using a strict allowlist of permitted fields.
        Guarantees that no private evaluation fields can leak.
        """
        return {
            "case_id": self.case_id,
            "observations": self.observations.model_dump()
        }
