"""
Observability and Telemetry Tracing for Automotive Digital Twin Diagnosis.
Captures workflow latencies, EIG entropy reduction, and review gate decisions.
Integrates with Langfuse if cloud credentials are provided; falls back to structured
local JSON logging for 100% offline reproducibility.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import uuid

from agents.state import DiagnosticReport
from model.schema import Observations


logger = logging.getLogger("automotive_twin.telemetry")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [TELEMETRY] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DiagnosticTracer:
    """
    Manages diagnostic trace lifecycle and telemetry metrics collection.
    """
    def __init__(self):
        self._trace_buffer: deque = deque(maxlen=200)
        self._langfuse_client = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pk and sk:
            try:
                from langfuse import Langfuse
                self._langfuse_client = Langfuse(public_key=pk, secret_key=sk, host=host)
                logger.info(f"Langfuse tracing enabled at {host}")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}. Falling back to local telemetry.")
                self._langfuse_client = None
        else:
            self._langfuse_client = None

    def record_trace(
        self,
        observations: Observations,
        report: DiagnosticReport,
        duration_ms: float,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Records a completed diagnostic run trace with performance and decision metrics.
        """
        trace_id = str(uuid.uuid4())
        follow_up = report.follow_up_question or {}
        trace_record = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id or report.case_id,
            "duration_ms": round(duration_ms, 2),
            "input_metrics": {
                "dtc_count": len(observations.dtc_codes),
                "symptom_count": len(observations.symptoms),
                "dtc_codes": observations.dtc_codes,
            },
            "output_metrics": {
                "top_fault_id": report.top_fault_id,
                "confidence_score": round(report.confidence_score, 4),
                "confidence_gap": round(report.confidence_gap, 4),
                "is_ambiguous": report.is_ambiguous,
                "insufficient_evidence": report.insufficient_evidence,
                "review_status": report.review_status,
                "citation_count": len(report.citations),
            },
            "adaptive_questioning": {
                "question_selected": bool(report.follow_up_question),
                "test_id": follow_up.get("test_id"),
                "eig_score": follow_up.get("eig_score"),
            },
        }

        # Store in rolling in-memory buffer
        self._trace_buffer.append(trace_record)

        # Log structured JSON telemetry line
        logger.info(json.dumps(trace_record))

        # Push to Langfuse if available
        if self._langfuse_client:
            try:
                self._langfuse_client.trace(
                    id=trace_id,
                    name="automotive_diagnosis_run",
                    metadata=trace_record,
                    input=observations.model_dump(),
                    output=report.model_dump(),
                )
            except Exception as e:
                logger.warning(f"Failed to push trace to Langfuse: {e}")

        return trace_record

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent diagnostic traces."""
        traces = list(self._trace_buffer)
        return traces[-limit:]


# Global singleton tracer instance
_global_tracer: Optional[DiagnosticTracer] = None


def get_tracer() -> DiagnosticTracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DiagnosticTracer()
    return _global_tracer
