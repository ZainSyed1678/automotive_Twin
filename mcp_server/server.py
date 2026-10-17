"""
Automotive Digital Twin MCP Server.
Exposes read-only diagnostic, simulation, and retrieval tools via the Python MCP SDK.
"""

from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from model.engine import FaultModel
from model.schema import Observations
from rag.store import KnowledgeBase, SimilarCaseIndex


def create_mcp_server(
    model: Optional[FaultModel] = None,
    kb: Optional[KnowledgeBase] = None,
    case_index: Optional[SimilarCaseIndex] = None,
) -> MCPServer:
    """
    Factory function to initialize the MCP server and register read-only tools.
    """
    fault_model = model or FaultModel()
    knowledge_base = kb or KnowledgeBase()
    similar_cases = case_index or SimilarCaseIndex()

    server = MCPServer(
        name="automotive_digital_twin",
        version="1.0.0",
        instructions="Read-only diagnostic digital twin tools for automotive fault investigation.",
    )

    @server.tool()
    def get_code_info(code: str) -> Dict[str, Any]:
        """
        Retrieve technical information, enabling criteria, and diagnostic steps for an OBD-II code.
        Example code: 'P0171' or 'P0101'.
        """
        if not code or not code.strip():
            return {"error": "Invalid input: 'code' must be a non-empty string."}

        res = knowledge_base.get_code_info(code)
        if not res:
            return {"error": f"No technical documentation found for DTC '{code.strip().upper()}'."}
        return res.model_dump()

    @server.tool()
    def search_knowledge(
        query: str,
        doc_type: Optional[str] = None,
        component_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search across technical repair manuals, OBD-II definitions, and TSBs.
        Query example: 'smoke test vacuum leak' or 'fuel rail pressure drop under load'.
        """
        if not query or not query.strip():
            return []

        results = knowledge_base.search(
            query=query,
            doc_type=doc_type,
            component_id=component_id,
            top_k=max(1, min(top_k, 10)),
        )
        return [r.model_dump() for r in results]

    @server.tool()
    def find_similar_cases(
        dtc_codes: Optional[List[str]] = None,
        symptoms: Optional[Dict[str, str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Search historical workshop development service cases matching the provided observations.
        Returns verified historical repair outcomes and matching evidence.
        """
        obs = Observations(
            dtc_codes=dtc_codes or [],
            symptoms=symptoms or {},
        )
        results = similar_cases.find_similar(obs, top_k=max(1, min(top_k, 10)))
        return [r.model_dump() for r in results]

    @server.tool()
    def simulate_fault(fault_id: str) -> Dict[str, Any]:
        """
        Simulate a fault in the digital twin and retrieve expected symptoms, DTC distributions,
        dominant physical behavior, and recommended distinguishing tests.
        Example fault_id: 'intake_vacuum_leak', 'fuel_pump_weak', 'maf_contamination'.
        """
        if not fault_id or not fault_id.strip():
            return {"error": "Invalid input: 'fault_id' must be a non-empty string."}

        try:
            sim = fault_model.simulate_fault(fault_id.strip())
            return sim.model_dump()
        except KeyError:
            return {"error": f"Fault '{fault_id}' not found in the digital twin catalogue."}

    @server.tool()
    def score_hypotheses(
        dtc_codes: Optional[List[str]] = None,
        symptoms: Optional[Dict[str, str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score candidate fault hypotheses using the digital twin likelihood scoring engine.
        Outputs ranked posterior probabilities, confidence gap, ambiguity flag, and evidence breakdown.
        """
        obs = Observations(
            dtc_codes=dtc_codes or [],
            symptoms=symptoms or {},
            notes=notes,
        )
        result = fault_model.score_hypotheses(obs)
        return result.model_dump()

    return server


# Default singleton instance for direct CLI or stdio execution
server = create_mcp_server()

if __name__ == "__main__":
    server.run(transport="stdio")
