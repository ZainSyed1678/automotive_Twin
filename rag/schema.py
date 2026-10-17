"""
Schemas for Knowledge Base Documents, Hybrid Search, and Similar-Case Retrieval.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    OBD_CODE = "obd_code"
    REPAIR_MANUAL = "repair_manual"
    TSB = "tsb"


class KnowledgeDocument(BaseModel):
    doc_id: str
    title: str
    doc_type: DocType
    component_id: Optional[str] = None
    related_dtcs: List[str] = Field(default_factory=list)
    summary: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    component_id: Optional[str] = None
    related_dtcs: List[str] = Field(default_factory=list)
    score: float
    snippet: str
    citation: str


class SimilarCaseResult(BaseModel):
    case_id: str
    similarity_score: float
    matching_dtcs: List[str]
    matching_symptoms: Dict[str, str]
    confirmed_fault: str
    confirmed_component: str
    difficulty: str
    repair_summary: str
