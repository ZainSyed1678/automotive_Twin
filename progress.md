# AI-Powered Automotive Digital Twin for Fault Diagnosis — Project Progress

## 1. Project Overview & Architecture Status
- **Goal**: Portfolio-grade AI-powered digital twin for simulation-driven vehicle fault diagnosis.
- **Domain**: Air-intake and fuel-control subsystem of a generic petrol vehicle.
- **Methodology**: Transparent Bayesian likelihood scoring (Good's weight of evidence / Bayes factor), simulation-driven hypothesis testing, synthetic case evaluation, hybrid RAG knowledge retrieval, standardized MCP tools, and LangGraph multi-node agent orchestration.
- **Guiding Principles**: Ponytail engineering (YAGNI, minimal dependencies, stdlib-first, transparent math over black boxes, rigorous test coverage).

---

## 2. Milestone Roadmap & Progress

| Milestone | Title | Status | Details |
|---|---|---|---|
| **M1** | **Fault Model & Scoring Engine** | **Completed** | 10 components, 12 faults, causal catalogue, log-likelihood ratio scoring, 6/6 tests passing, demo runner verified. |
| **M2** | **Diagnostic Case Generator** | **Completed** | 200 synthetic cases (140 dev, 60 eval), 5 difficulty tiers, 0 fingerprint collisions, allowlist agent isolation, 17/17 tests passing. |
| **M3** | **Knowledge Base & MCP Tools** | **Completed** | 24 technical documents (OBD/Manuals/TSBs), TF-IDF hybrid search, isolated similar-case search, Python MCP server with 5 read-only tools, 28/28 tests passing. |
| **M4** | **Diagnostic Agent Core & Baselines** | **Completed** | LangGraph StateGraph (5 nodes), 3 diagnostic baselines (LLM-Only, RAG-Only, Full System), 140-case development evaluation, 35/35 tests passing. |
| **M5** | **Adaptive Questioning & Review** | *Next* | Information-gain question selection, evidence critique, review node. |
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
- **24 Technical Documents**: 12 OBD-II code sheets, 6 repair manual procedures, 6 TSBs.
- **Hybrid Search Index**: TF-IDF cosine similarity + structural metadata filtering.
- **Similar Case Search**: Strict development-only index, Jaccard + symptom overlap.
- **Python MCP Server (`mcp_server/server.py`)**: Exposes 5 read-only tools via `MCPServer` (`get_code_info`, `search_knowledge`, `find_similar_cases`, `simulate_fault`, `score_hypotheses`).

---

## 6. Milestone 4 — Diagnostic Agent Core & Comparative Baselines

### LangGraph Diagnostic Architecture (`agents/workflow.py`, `agents/nodes.py`)
- **StateGraph**: Coordinates 5 logical nodes:
  1. `intake`: Normalizes incoming DTCs, sensor readings, and symptoms.
  2. `retrieval`: Queries `KnowledgeBase` and `SimilarCaseIndex` to collect evidence and verifiable citations.
  3. `hypothesis_generation`: Gathers candidate faults matching observed codes, symptoms, and priors.
  4. `simulation_scoring`: Evaluates candidate hypotheses against the causal digital twin using Bayesian likelihood scoring; runs `simulate_fault` on top candidates.
  5. `reporting`: Compiles a structured, evidence-backed `DiagnosticReport` with explicit citations (`[DOC-...]`, `[CASE-...]`), confidence margins, ambiguity flags, and recommended next tests.

### Comparative Baseline Implementations (`eval/baselines.py`)
1. **LLM-Only**: Diagnostic reasoning based strictly on raw observations using zero-shot heuristic pattern matching (no RAG retrieval, no digital twin simulation).
2. **RAG-Only**: Hypotheses scored using retrieved technical documentation relevance and historical similar case frequencies (no causal digital twin simulation).
3. **Full System**: End-to-end integration of technical retrieval + historical cases + causal digital twin simulation and Bayesian likelihood ratio scoring.

### Initial Comparative Evaluation Results (140 Development Cases)
Evaluated via `python -m eval.evaluate` on `data/cases_dev.json`:

| Metric | LLM-Only | RAG-Only | Full System (Digital Twin) |
|---|---|---|---|
| **Top-1 Diagnostic Accuracy** | 53.2% | 81.0% | 74.6% |
| **Top-3 Diagnostic Accuracy** | 58.7% | 92.1% | 81.0% |
| **Insufficient Evidence Handling** | 50.0% | 85.7% | **100.0%** |
| **Ambiguity Detection Rate** | 40.0% | 34.3% | 32.9% |
| **Avg Citations per Case** | 0.00 | 5.07 | **5.44** |
| **Unsupported Claim Rate** | 100.0% | 2.4% | **0.0%** |

### Analysis of Baseline Insights
- **Unsupported Claims Eliminated**: The Full System achieved a **0.0% unsupported claim rate** (every single assertion was backed by an explicit technical citation or digital twin simulation proof), compared to 100% unsupported claims in LLM-Only.
- **Perfect Insufficient Evidence Abstention**: The Full System correctly abstained on **100.0%** of unprovable/sparse cases, preventing the dangerous hallucinations observed in pure LLM reasoning.
- **RAG vs Physics-Twin Complementarity**: RAG-Only excels at historical case pattern-matching (81.0% Top-1), while the Full System introduces strict physics constraints (penalizing contradictions and enforcing physical diagnostic consistency).

### Test Verification (`tests/test_agent_and_eval.py`)
- 35 total tests passing across M1, M2, M3, and M4 in 8.09s.

---

## 7. Milestone 5 — Adaptive Questioning & Evidence Review Gate

### Mathematical Formulation of Expected Information Gain (EIG) (`agents/eig.py`)
When candidate faults compete closely (`is_ambiguous == True` or `confidence_gap < 0.25`), the diagnostic agent determines the most informative physical test to perform instead of guessing:

1. **Prior Shannon Entropy**:
   $$H(F) = -\sum_{i=1}^K P(F_i) \log_2 P(F_i)$$
2. **Marginal Predictive Probability for Test Variable $X$**:
   $$P(X = v) = \sum_{i=1}^K P(F_i) \cdot P(X = v \mid F_i)$$
3. **Posterior Conditional Distribution**:
   $$P(F_i \mid X = v) = \frac{P(F_i) \cdot P(X = v \mid F_i)}{P(X = v)}$$
4. **Conditional Entropy**:
   $$H(F \mid X = v) = -\sum_{i=1}^K P(F_i \mid X = v) \log_2 P(F_i \mid X = v)$$
5. **Expected Information Gain (Mutual Information $I(F; X)$)**:
   $$\text{EIG}(X) = H(F) - \sum_{v \in \mathcal{V}_X} P(X = v) \cdot H(F \mid X = v)$$

The engine selects the candidate test $X^* = \arg\max_X \text{EIG}(X)$ that has not yet been observed.

### Actionable Test Catalogue (`agents/eig.py`)
Codified 11 physical inspection and scanner test procedures with baseline distributions, technician-facing questions, and discrimination targets:
- `fuel_pressure` (Mechanical gauge test: normal / low / high)
- `fuel_trim_at_high_rpm` (2500 RPM steady state: normalizes / worsens / unchanged)
- `audible_hiss` (Plenum stethoscope check: present / absent)
- `misfire_pattern` (Scanner counter: cylinder_specific / random / none)
- `maf_reading` (WOT g/s check: normal / underreporting / overreporting)
- `map_reading_key_on_engine_off` (KOEO baro check: normal / inaccurate)
- `o2_sensor_switching_rate` (Oscilloscope 2000 RPM check: normal / sluggish)
- `throttle_sweep_test` (Potentiometer sweep: normal / dropouts_at_low_throttle)
- `connector_pin_inspection` (Visual & resistance check: clean / corroded_green)
- `coolant_temp_scan` (Cold ambient sanity check: normal / abnormally_low)
- `fuel_pressure_bleed_down` (10-min rail retention test: holds_steady / rapid_bleed_down)

### Self-Review Verification Gate (`agents/review.py`)
Before any report is emitted, an automated review audit validates three safety criteria:
1. **Scope Boundary**: Confirms all candidate faults and components belong strictly to the supported subsystem (`air_intake_and_fuel_control` on a generic petrol car).
2. **Evidence Grounding**: Verifies that every asserted symptom and DTC match exists in the raw input observations. Flags ungrounded claims.
3. **Epistemic Humility**: Ensures insufficient evidence cases suppress premature confidence and ambiguous cases warn technicians against unverified parts replacement.

### LangGraph Workflow Integration (`agents/workflow.py`, `agents/nodes.py`)
- Expanded StateGraph from 5 to 7 logical nodes:
  $$\text{START} \to \text{intake} \to \text{retrieval} \to \text{hypothesis\_generation} \to \text{simulation\_scoring} \to \text{question\_selection} \to \text{review} \to \text{reporting} \to \text{END}$$
- Implemented `run_interactive_diagnosis(initial_observations, answer_callback, max_turns=3)` enabling multi-turn conversational disambiguation.

### Test Verification (`tests/test_adaptive_questioning.py`)
- 9 dedicated unit and scenario tests covering:
  - Analytical Shannon entropy calculations (0.0 bits for deterministic, 1.0 bit for 2-state, 2.0 bits for 4-state).
  - P0171 lean ambiguity test: verifies EIG prioritizes `fuel_pressure` ($\text{EIG} = 0.6417$ bits) over irrelevant tests.
  - Stopping condition: halts follow-up questions when confidence and gap are high.
  - Review gate verification: passes valid reports, flags out-of-scope components, and detects ungrounded evidence assertions.
  - Multi-turn interactive disambiguation: resolves ambiguous P0171 case in 2 turns when fuel pressure is reported as low.
- Total test suite: **44 passed in 7.34s**.

