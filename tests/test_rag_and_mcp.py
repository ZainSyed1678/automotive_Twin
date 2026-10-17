"""
Automated test suite for M3: Knowledge Base, Hybrid RAG, Similar Cases, and MCP Server.
"""

import pytest
from rag import DocType, KnowledgeBase, SimilarCaseIndex
from model.schema import Observations
from mcp_server import create_mcp_server


# -----------------------------------------------------------------------------
# 1. Knowledge Base Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def kb():
    return KnowledgeBase()


def test_knowledge_base_loading_and_counts(kb: KnowledgeBase):
    """Verify all 24 technical documents are loaded and indexed correctly."""
    assert len(kb.documents) == 24
    obd_docs = [d for d in kb.documents if d.doc_type == DocType.OBD_CODE]
    manual_docs = [d for d in kb.documents if d.doc_type == DocType.REPAIR_MANUAL]
    tsb_docs = [d for d in kb.documents if d.doc_type == DocType.TSB]

    assert len(obd_docs) == 12, f"Expected 12 OBD codes, got {len(obd_docs)}"
    assert len(manual_docs) == 6, f"Expected 6 repair manuals, got {len(manual_docs)}"
    assert len(tsb_docs) == 6, f"Expected 6 TSBs, got {len(tsb_docs)}"

    for doc in kb.documents:
        assert doc.doc_id.startswith("DOC-")
        assert len(doc.title) > 5
        assert len(doc.content) > 50


def test_get_code_info(kb: KnowledgeBase):
    """Verify direct lookup for generic OBD-II fault codes."""
    res_p0171 = kb.get_code_info("P0171")
    assert res_p0171 is not None
    assert res_p0171.doc_id == "DOC-OBD-P0171"
    assert "System Too Lean" in res_p0171.title
    assert "P0171" in res_p0171.related_dtcs

    # Case-insensitive lookup
    res_lower = kb.get_code_info("p0101")
    assert res_lower is not None
    assert res_lower.doc_id == "DOC-OBD-P0101"

    # Unknown code
    assert kb.get_code_info("P9999") is None


def test_hybrid_search(kb: KnowledgeBase):
    """Verify hybrid search returns relevant documents and respects metadata filters."""
    # 1. Smoke test query should rank DOC-MAN-SMOKE-TEST at or near top
    smoke_results = kb.search("smoke machine intake leak test", top_k=3)
    assert len(smoke_results) > 0
    top_doc_ids = [r.doc_id for r in smoke_results]
    assert "DOC-MAN-SMOKE-TEST" in top_doc_ids

    # 2. Oscilloscope TPS sweep query
    tps_results = kb.search("oscilloscope potentiometer waveform sweep throttle", top_k=3)
    assert len(tps_results) > 0
    assert tps_results[0].doc_id == "DOC-MAN-TPS-OSC"

    # 3. Filter by doc_type='tsb'
    tsb_results = kb.search("fuel trim at 2500 rpm", doc_type="tsb", top_k=5)
    assert len(tsb_results) > 0
    for r in tsb_results:
        assert r.doc_type == "tsb"
    assert "DOC-TSB-2023-01" in [r.doc_id for r in tsb_results]


# -----------------------------------------------------------------------------
# 2. Similar Case Search & Isolation Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def case_index():
    return SimilarCaseIndex()


def test_similar_case_index_security():
    """Verify that SimilarCaseIndex refuses to load held-out evaluation cases."""
    with pytest.raises(PermissionError):
        SimilarCaseIndex(dev_cases_path="data/cases_eval.json")


def test_find_similar_cases(case_index: SimilarCaseIndex):
    """Verify finding similar historical cases from the development dataset."""
    obs = Observations(
        dtc_codes=["P0171"],
        symptoms={"fuel_pressure": "normal", "idle_quality": "rough"}
    )
    similar = case_index.find_similar(obs, top_k=3)
    assert len(similar) == 3

    for item in similar:
        assert item.case_id.startswith("CASE-")
        assert item.similarity_score > 0.0
        assert item.confirmed_fault != ""
        assert len(item.repair_summary) > 10


# -----------------------------------------------------------------------------
# 3. MCP Server Tool Integration Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def mcp():
    return create_mcp_server()


@pytest.mark.asyncio
async def test_mcp_tool_registration(mcp):
    """Verify all 5 required read-only tools are registered on the MCP server."""
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    expected_tools = {
        "get_code_info",
        "search_knowledge",
        "find_similar_cases",
        "simulate_fault",
        "score_hypotheses",
    }
    assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"


@pytest.mark.asyncio
async def test_mcp_get_code_info(mcp):
    """Test calling get_code_info tool via MCP interface."""
    res = await mcp.call_tool("get_code_info", {"code": "P0171"})
    assert res.is_error is False
    assert res.structured_content is not None
    data = res.structured_content.get("result", res.structured_content)
    assert data.get("doc_id") == "DOC-OBD-P0171"


@pytest.mark.asyncio
async def test_mcp_search_knowledge(mcp):
    """Test calling search_knowledge tool via MCP interface."""
    res = await mcp.call_tool(
        "search_knowledge",
        {"query": "fuel pressure drop under load", "top_k": 2}
    )
    assert res.is_error is False
    assert res.structured_content is not None
    results = res.structured_content.get("result", [])
    assert len(results) > 0


@pytest.mark.asyncio
async def test_mcp_find_similar_cases(mcp):
    """Test calling find_similar_cases tool via MCP interface."""
    res = await mcp.call_tool(
        "find_similar_cases",
        {"dtc_codes": ["P0171"], "symptoms": {"idle_quality": "rough"}, "top_k": 3}
    )
    assert res.is_error is False
    assert res.structured_content is not None
    cases = res.structured_content.get("result", [])
    assert len(cases) == 3
    for c in cases:
        assert "case_id" in c
        assert "confirmed_fault" in c


@pytest.mark.asyncio
async def test_mcp_simulate_fault(mcp):
    """Test calling simulate_fault tool via MCP interface."""
    # 1. Valid fault
    res = await mcp.call_tool("simulate_fault", {"fault_id": "intake_vacuum_leak"})
    assert res.is_error is False
    data = res.structured_content.get("result", res.structured_content)
    assert data.get("fault_id") == "intake_vacuum_leak"
    assert "P0171" in data.get("expected_dtcs", {})

    # 2. Unknown fault returns error dict
    res_err = await mcp.call_tool("simulate_fault", {"fault_id": "unknown_component_burn"})
    err_data = res_err.structured_content.get("result", res_err.structured_content)
    assert "error" in err_data


@pytest.mark.asyncio
async def test_mcp_score_hypotheses(mcp):
    """Test calling score_hypotheses tool via MCP interface."""
    res = await mcp.call_tool(
        "score_hypotheses",
        {
            "dtc_codes": ["P0171"],
            "symptoms": {
                "fuel_pressure": "normal",
                "fuel_trim_at_high_rpm": "normalizes",
                "audible_hiss": "present"
            }
        }
    )
    assert res.is_error is False
    data = res.structured_content.get("result", res.structured_content)
    assert data.get("top_fault_id") == "intake_vacuum_leak"
    assert len(data.get("ranked_hypotheses", [])) == 12
