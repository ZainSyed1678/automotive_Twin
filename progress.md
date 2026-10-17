# AI-Powered Automotive Digital Twin for Fault Diagnosis — Project Progress

## 1. Project Overview & Architecture Status
- **Goal**: Portfolio-grade AI-powered digital twin for simulation-driven vehicle fault diagnosis.
- **Domain**: Air-intake and fuel-control subsystem of a generic petrol vehicle.
- **Methodology**: Transparent Bayesian likelihood scoring (Good's weight of evidence / Bayes factor), simulation-driven hypothesis testing, synthetic case evaluation, hybrid RAG knowledge retrieval, and standardized MCP tools.
- **Guiding Principles**: Ponytail engineering (YAGNI, minimal dependencies, stdlib-first, transparent math over black boxes, rigorous test coverage).

---

## 2. Milestone Roadmap & Progress

| Milestone | Title | Status | Details |
|---|---|---|---|
| **M1** | **Fault Model & Scoring Engine** | **Completed** | 10 components, 12 faults, causal catalogue, log-likelihood ratio scoring, 6/6 tests passing, demo runner verified. |
| **M2** | **Diagnostic Case Generator** | **Completed** | 200 synthetic cases (140 dev, 60 eval), 5 difficulty tiers, 0 fingerprint collisions, allowlist agent isolation, 17/17 tests passing. |
| **M3** | **Knowledge Base & MCP Tools** | **Completed** | 24 technical documents (OBD/Manuals/TSBs), TF-IDF hybrid search, isolated similar-case search, Python MCP server with 5 read-only tools, 28/28 tests passing. |
| **M4** | **Diagnostic Agent Core** | *Next* | LangGraph orchestration (intake, retrieve, hypothesize, score, report), LLM-only & RAG-only baselines. |
| **M5** | **Adaptive Questioning & Review** | Pending | Information-gain question selection, evidence critique, insufficient-evidence handling. |
| **M6** | **Interface & Observability** | Pending | FastAPI backend, Streamlit diagnostic UI, Docker containerization. |
| **M7** | **Evaluation & Documentation** | Pending | Held-out benchmarking across 3 modes, calibration scoring, honest audit. |

---

## 3. Milestone 1 — Fault Model & Scoring Engine

### Architecture & Schemas (`model/schema.py`)
- **`Component`**: Models physical vehicle subsystem parts with id, name, and description.
- **`FaultDefinition`**: Defines failure modes with prior probability, expected DTCs with likelihoods $P(\text{DTC} \mid \text{Fault})$, expected symptom states with likelihoods $P(\text{state} \mid \text{Fault})$, and distinguishing physical tests.
- **`Observations`**: Carries agent/technician observable inputs (DTC list, symptom dict, notes).
- **`SimulationResult`**: Predicts dominant physical symptoms and DTC distributions for any known fault.
- **`HypothesisScore` & `DiagnosticResult`**: Outputs ranked posterior probabilities, confidence gap, ambiguity flag, insufficient evidence flag, and evidence match/contradiction tracking.

### Causal Catalogue (`model/catalogue.yaml`)
- **10 Core Components**: MAF sensor, MAP sensor, TPS, Upstream O2 sensor, ECT sensor, Fuel injector, Fuel pump, Intake vacuum line, Wiring harness, ECU.
- **12 Representative Faults**: `maf_contamination`, `intake_vacuum_leak`, `map_sensor_drift`, `o2_sensor_slow_response`, `fuel_injector_clogged`, `fuel_pump_weak`, `connector_corrosion`, `wiring_open_circuit`, `wiring_short_circuit`, `coolant_sensor_stuck_low`, `throttle_sensor_drift`, `fuel_pressure_regulator_stuck_open`.

### Scoring Mathematics (`model/engine.py`)
- Uses transparent **Log-Likelihood Ratio (Bayes Factor / Weight of Evidence)** scoring:
  $$\text{Score}(F) = \log P(F) + \sum_{o \in \text{Obs}} \log\left(\frac{P(o \mid F)}{P_{\text{baseline}}(o)}\right)$$
- Missing evidence evaluates to $W = 0.0$ (neutral).
- Contradictory evidence applies an $\epsilon$ penalty floor.
- Ambiguity flagged when $(\text{Top 1} - \text{Top 2}) < 0.15$.
- Insufficient evidence flagged when no observations exist or top candidate posterior $< 0.25$.

---

## 4. Milestone 2 — Diagnostic Case Generator

### Architecture & Ground-Truth Isolation (`casegen/schema.py`)
- **`GroundTruthCase`**: Contains `case_id`, `true_fault_id`, `difficulty`, `observations: Observations`, `generation_metadata`.
- **`AgentCase`**: Pydantic model with `ConfigDict(extra="forbid")` containing strictly `case_id` and `observations`.
- **Allowlist Serialization**: Evaluated files written to `data/agent_cases/` are filtered through an explicit field allowlist (`{"case_id", "observations"}`). Automated tests verify that `true_fault_id`, `difficulty`, and metadata never appear in agent-visible files.

### Dataset Generation & Stratification (`casegen/generator.py`, `casegen/split.py`)
- **Total Cases**: 200 cases generated using an isolated seeded generator (`seed=42`).
- **Dataset Split**:
  - Development Set: 140 cases (70.0%)
  - Held-out Evaluation Set: 60 cases (30.0%)
- **Stratified Difficulty Distribution (Exact 70% Dev / 30% Eval per tier)**:
  - Straightforward: 70 total (49 Dev, 21 Eval)
  - Incomplete: 50 total (35 Dev, 15 Eval)
  - Ambiguous: 40 total (28 Dev, 12 Eval)
  - Misleading: 20 total (14 Dev, 6 Eval)
  - Insufficient Evidence: 20 total (14 Dev, 6 Eval)
- **Balanced Fault Representation**: All 12 faults are represented across both development and evaluation sets (16 to 17 cases per fault).

### Anti-Leakage & Fingerprinting Validation
- **Canonical Observation Fingerprint**:
  $$\text{Fingerprint} = \text{SHA256}(\text{JSON}(\text{sorted\_dtcs}, \text{sorted\_symptoms}))$$
  - Order-invariant: permutation of DTC or symptom keys produces the exact same fingerprint.
  - Excludes case IDs, notes, and ground-truth metadata.
- **Fingerprint Overlap Check**:
  $$\text{fingerprints}(\text{dev}) \cap \text{fingerprints}(\text{eval}) = \emptyset$$
  - Exactly 140 unique fingerprints in dev.
  - Exactly 60 unique fingerprints in eval.
  - Exactly **0 collisions** between dev and eval.

---

## 5. Milestone 3 — Knowledge Base & MCP Tools

### Technical Document Store (`rag/documents.yaml`, `rag/schema.py`)
- **24 Technical Documents**:
  - **12 OBD-II Code Reference Guides**: `P0171`, `P0101`, `P0106`, `P0133`, `P0087`, `P0117`, `P0121`, `P0507`, `P0300`, `P0102`, `P0103`, `P0100`.
  - **6 OEM-Style Repair Manual Procedures**: Smoke machine testing, fuel rail static/dynamic testing, MAF WOT flow check, TPS oscilloscope sweep, ECT bench resistance, harness terminal drag/voltage drop.
  - **6 Technical Service Bulletins (TSBs)**: Idle vs 2500 RPM trim diagnostic rule, MAF silicone vapor contamination, terminal fretting corrosion, ethanol varnish injector clogging, O2 sensor delay hunting, bellhousing ground reference shift.

### Hybrid Search & Similar-Case Retrieval Engine (`rag/store.py`)
- **`KnowledgeBase`**:
  - `get_code_info(code)`: Fast direct O(1) lookup returning structured definitions and diagnostic procedures.
  - `search(query, doc_type, component_id, top_k)`: TF-IDF vector space retrieval with sublinear term frequency scaling, metadata keyword boosts (+0.25 on related DTC hits), and structural category filtering.
  - All results return verifiable citation handles (e.g. `[DOC-TSB-2023-01]`) and concise snippets.
- **`SimilarCaseIndex`**:
  - Indexes strictly from `data/cases_dev.json`.
  - **Security Guard**: Raises `PermissionError` if pointed to evaluation files.
  - `find_similar(observations, top_k)`: Computes Jaccard DTC similarity and symptom key-value overlap to find nearest historical workshop cases, returning verified historical resolution summaries without exposing generator internals.

### Python MCP Server (`mcp_server/server.py`)
- Implemented with modern Python MCP SDK (`mcp 2.2.0`, `MCPServer`).
- Exposes 5 read-only tools:
  1. `get_code_info(code: str) -> dict`
  2. `search_knowledge(query: str, doc_type: Optional[str], component_id: Optional[str], top_k: int) -> list[dict]`
  3. `find_similar_cases(dtc_codes: list[str], symptoms: dict, top_k: int) -> list[dict]`
  4. `simulate_fault(fault_id: str) -> dict`
  5. `score_hypotheses(dtc_codes: list[str], symptoms: dict, notes: str) -> dict`
- Validates external inputs, provides error dictionaries for unknown codes/faults, and runs over standard stdio transport.

### Test Verification (`tests/test_rag_and_mcp.py`)
- 28 total tests passing across M1, M2, and M3 in 7.43s:
  - `test_knowledge_base_loading_and_counts`: 24 documents loaded.
  - `test_get_code_info`: Direct OBD lookup verified.
  - `test_hybrid_search`: Natural language queries + category filtering verified.
  - `test_similar_case_index_security`: Evaluation dataset access forbidden.
  - `test_find_similar_cases`: Case matching and repair summaries verified.
  - `test_mcp_tool_registration`: All 5 tools registered.
  - `test_mcp_get_code_info`, `test_mcp_search_knowledge`, `test_mcp_find_similar_cases`, `test_mcp_simulate_fault`, `test_mcp_score_hypotheses`: Full tool execution verified.
