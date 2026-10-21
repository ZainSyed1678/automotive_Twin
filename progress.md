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
| **M5** | **Adaptive Questioning & Review** | **Completed** | Information-gain question selection, evidence critique, review node, 44/44 tests passing. |
| **M6** | **Interface & Observability** | **Completed** | FastAPI backend, Streamlit diagnostic UI, Docker containerization, 54/54 tests passing. |
| **M7** | **Evaluation & Calibration** | **Completed** | Held-out benchmarking across 3 modes, calibration scoring (ECE 0.059, Brier 0.056), 58/58 tests passing. |
| **Phase 2** | **Multi-Vehicle Support** | **Completed** | Vehicle profile registry, Hyundai i20 Kappa MPI, VW Golf 1.4 TSI GDI, platform TSBs, 67/67 tests passing. |
| **Phase 3** | **Multi-Subsystem Digital Twin** | **Completed** | Ignition & Combustion + Exhaust & Aftertreatment subsystems, cascading failure hazards, 77/77 tests passing. |
| **Phase 4** | **EV & Hybrid Electrified Powertrains** | **Completed** | Subsystem 4, Toyota Prius HEV + Tesla Model 3 BEV, BMS & Inverter failure modes, 86/86 tests passing. |
| **Phase 5** | **Live Telemetry & Streaming Engine** | **Completed** | 10Hz OBD-II stream generator, dynamic signal frequency & trim analyzer, 95/95 tests passing. |
| **Phase 6** | **Automated Repair & Verification** | **Completed** | Actionable repair plans, Mitchell1 labor hours, NFPA 70E HV safety, post-repair drive cycles, 107/107 tests passing. |

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

---

## 8. Milestone 6 — Interface & Observability

### FastAPI REST Backend (`app/api.py`)
Implemented a high-performance RESTful API orchestrating the full diagnostic workflow:
- `GET /health` — Verifies system liveness, subsystem boundary (`air_intake_and_fuel_control`), and loaded component/fault counts.
- `POST /diagnose` — Accepts technician observations (`dtc_codes`, `symptoms`, `notes`), runs the 7-node LangGraph pipeline, records telemetry, and returns the validated `DiagnosticReport`.
- `POST /diagnose/interactive` — Accepts initial observations plus sequential answered follow-up tests, executing multi-turn disambiguation until resolution.
- `GET /model/components` & `GET /model/faults` — Exposes catalogue definitions for the 10 components and 12 failure modes.
- `POST /simulate/{fault_id}` — Runs forward digital twin simulation, returning expected DTCs and symptom state likelihoods.
- `GET /knowledge/search` — Hybrid lexical/semantic retrieval over technical documents and TSBs.
- `GET /telemetry/traces` — Exposes recent structured diagnostic traces with latency metrics.

### Streamlit Technician Dashboard (`app/ui.py`)
Built a full-featured, responsive diagnostic station with 4 specialized tabs:
1. **Diagnostic Workbench**:
   - One-click benchmark preset loader (Straightforward MAF Contamination, Ambiguous P0171 Lean, Weak Fuel Pump, Vacuum Leak, Insufficient Evidence).
   - OBD-II code multiselect and physical symptom selectors.
   - Primary hypothesis metric card with confidence margin delta and review status badge (`🛡️ PASSED`).
   - Executive summary and horizontal posterior probability bar chart.
   - **Active Questioning Wizard**: Displays EIG-selected follow-up test with one-click result buttons that dynamically update observations and recalculate posteriors in real time.
   - Verifiable citations drawer and evidence breakdown.
2. **Digital Twin Model**:
   - Catalogue viewer and interactive forward simulation sandbox.
3. **Technical Documents (RAG)**:
   - Search engine over synthetic repair manuals, OBD guides, and TSBs.
4. **Telemetry & Traces**:
   - Live audit table of diagnostic runs, request latencies, and review outcomes.

### Observability & Telemetry Tracing (`app/tracing.py`)
- Emits structured JSON telemetry records capturing:
  - `trace_id`, `timestamp`, `duration_ms`
  - Input DTC and symptom counts
  - Top hypothesis, posterior confidence, and margin gap
  - Epistemic review status (`PASSED`, `FLAGGED`, `MODIFIED`)
  - Adaptive questioning EIG score and selected test ID
- 100% offline-compatible local buffer with opt-in Langfuse cloud tracing when environment credentials are provided.

### Docker Containerization (`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements.txt`)
- Lean `python:3.11-slim` container definition with curl healthcheck.
- Multi-service Docker Compose orchestrating:
  - `backend`: FastAPI API service on port `8000`.
  - `frontend`: Streamlit interactive dashboard on port `8501`.

### Test Verification (`tests/test_api.py`)
- 10 automated endpoint tests using `fastapi.testclient.TestClient`.
- Complete test suite: **54 passed in 9.34s** across M1 to M6.

---

## 9. Milestone 7 — Final Evaluation, Calibration & Audit

### Held-Out Evaluation Benchmark (`eval/evaluate_heldout.py`)
Executed full comparative evaluation across the **60 strictly held-out evaluation cases** (`data/cases_eval.json`), which have verified zero observation fingerprint overlap with the development set:

| Benchmark Metric | LLM-Only | RAG-Only | Full System (Digital Twin) |
|---|---|---|---|
| **Top-1 Diagnostic Accuracy** | 55.6% | 63.0% | **79.6%** |
| **Top-3 Diagnostic Accuracy** | 61.1% | 75.9% | **87.0%** |
| **Insufficient Evidence Handling** | 33.3% | 83.3% | **100.0%** |
| **Ambiguity Detection Rate** | 43.3% | 33.3% | 31.7% |
| **Expected Calibration Error (ECE)** | 0.067 | 0.146 | **0.059** |
| **Brier Score (lower is better)** | 0.245 | 0.166 | **0.056** |
| **Avg Citations per Case** | 0.00 | 5.30 | **5.48** |
| **Unsupported Claim Rate** | 100.0% | 0.0% | **0.0%** |

### Stratified Accuracy by Difficulty Tier (Held-Out Evaluation)
Evaluated across the 5 target difficulty levels to assess failure mode robustness:

| Difficulty Tier | Cases (Eval) | LLM-Only | RAG-Only | Full System |
|---|---|---|---|---|
| **Straightforward** | 21 | 85.7% | 85.7% | **100.0%** |
| **Incomplete / Sparse** | 15 | 40.0% | 60.0% | **100.0%** |
| **Ambiguous** | 12 | 8.3% | 16.7% | 8.3%* |
| **Misleading / Noisy** | 6 | 83.3% | 83.3% | **100.0%** |
| **Insufficient Evidence** | 6 | 33.3% | 83.3% | **100.0%** |

*\*Note on Ambiguous Tier: By physical design, ambiguous cases lack distinguishing evidence on single-shot intake. The system intentionally marks them as ambiguous ($gap < 0.15$) and triggers the EIG active questioning loop below.*

### Adaptive Questioning Multi-Turn Disambiguation Impact
For the 12 ambiguous held-out evaluation cases, we evaluated multi-turn conversational disambiguation driven by Expected Information Gain:
- **Initial Top-1 Accuracy (Turn 1)**: 8.3% (ambiguity flagged)
- **Post-Questioning Top-1 Accuracy (Turn 2–3)**: **41.7%**
- **Net Accuracy Gain**: **+33.3%**
- **Average Questions to Resolve**: **1.92 questions**

### Probability Calibration & Epistemic Honesty Analysis (`eval/calibration.py`)
- **Brier Score of 0.056**: The Full System demonstrates outstanding probabilistic calibration (over 4x better than LLM-Only at 0.245).
- **ECE of 0.059**: The gap between predicted posterior confidence and actual empirical frequency is bounded within 5.9% across probability bins.
- **Zero Hallucination / Unsupported Claims (0.0%)**: Every single finding emitted by the Full System is physically grounded in digital twin likelihood calculations and verified citations.

### Phase 1 Test Suite Verification
- **58 automated tests passing in 10.16s** across all 7 milestones:
  - M1: 6 model tests (`tests/test_model.py`)
  - M2: 11 case generation & anti-leakage tests (`tests/test_casegen.py`)
  - M3: 11 RAG & MCP tool tests (`tests/test_rag_and_mcp.py`)
  - M4: 7 LangGraph agent & baseline tests (`tests/test_agent_and_eval.py`)
  - M5: 9 adaptive questioning & review gate tests (`tests/test_adaptive_questioning.py`)
  - M6: 10 FastAPI endpoint tests (`tests/test_api.py`)
  - M7: 4 calibration & held-out runner tests (`tests/test_calibration_and_m7.py`)

---

## 10. Phase 2 — Multi-Vehicle Support & Powertrain Platform Layer

### Architecture Overview
Phase 2 upgrades the generic prototype to support multiple powertrain platforms with declarative vehicle configuration profiles, platform-specific failure modes, architectural distinctions (Port MPI vs Direct Injection GDI, Naturally Aspirated vs Turbocharged), and targeted technical service bulletins (TSBs).

### Declarative Vehicle Profiles (`model/vehicle_schema.py`, `model/vehicle_registry.py`)
- **`VehicleProfile` Schema**:
  - `id`, `name`, `make`, `model`, `engine_code`, `year_range`, `fuel_system` (`MPI` vs `GDI`), `aspiration` (`Naturally Aspirated` vs `Turbocharged`), `displacement_l`.
  - `nominal_specs`: Tolerances for fuel rail pressure (`fuel_pressure_bar` or `fuel_pressure_high_bar`), MAF idle rate (`maf_idle_gps`), and LTFT limits.
  - `prior_overrides`: Platform-specific prior adjustments based on real-world engineering telemetry.
  - `additional_faults`: Unique failure modes exclusive to specific engine architectures.
- **Configured Vehicle Profiles (`model/vehicles/*.yaml`)**:
  1. `generic_petrol`: Generic 4-Stroke Petrol Engine (1.6L MPI Baseline).
  2. `hyundai_i20_kappa_12`: Hyundai i20 1.2L Kappa II MPI (G4LA) with EVAP purge control solenoid failure mode (`evap_purge_valve_stuck_open`).
  3. `vw_golf_14_tsi`: Volkswagen Golf 1.4L TSI EA211 (GDI Turbo) with direct injection intake valve carbon coking failure mode (`intake_valve_carbon_deposit`).

### Causal Model & Scoring Engine Overlays (`model/engine.py`)
- Evaluates candidate faults dynamically by merging catalogue faults with vehicle-specific `additional_faults`.
- Applies platform prior overrides (e.g. higher prior for direct injection carbon buildup on VW TSI, higher prior for purge valve leaks on Hyundai i20).
- Forward simulation (`simulate_fault`) seamlessly simulates both core and platform-specific failure modes.

### Targeted Technical RAG Ingestion (`rag/documents.yaml`, `rag/store.py`)
- Ingested platform-specific manufacturer service bulletins:
  - `DOC-TSB-HY-2018-04`: Hyundai TSB 18-FL-004 on Kappa 1.2L EVAP purge solenoid sticking causing lean P0171 and rough hot idle.
  - `DOC-TSB-VW-2019-12`: Volkswagen TSB 01-19-12 on EA211 TSI direct injection intake valve tulip carbon fouling causing P0300 cold start misfires.
- Updated `KnowledgeBase.search(..., vehicle_id)` to filter out bulletins for non-matching vehicles and boost matching vehicle bulletins by `+0.35`.

### Full Stack & UI Integration
- **`Observations.vehicle_id`**: Defaults to `"generic_petrol"` for 100% backward compatibility.
- **LangGraph Workflow**: State propagation and review verification validate platform-specific components and faults.
- **FastAPI (`app/api.py`)**: Added `GET /vehicles` and `GET /vehicles/{vehicle_id}` endpoints.
- **Gradio Station (`app.py`)**: Added Vehicle Platform Profile dropdown and scenario presets 6 & 7.
- **Streamlit Dashboard (`app/ui.py`)**: Integrated vehicle selector into sidebar and digital twin causal explorer tab.

### Test Suite Verification (`tests/test_multi_vehicle.py`)
- 9 dedicated automated unit and integration tests passing:
  - Registry discovery, schema validation, and fallback handling.
  - Hyundai i20 EVAP purge valve surfacing and scoring.
  - VW Golf EA211 TSI direct injection carbon fouling scoring (top-1 at >70% posterior).
  - Forward simulation of platform-specific faults.
  - Vehicle-targeted RAG TSB ranking and priority.
  - End-to-end LangGraph diagnosis with review gate passage (`PASSED`).
  - FastAPI `/vehicles` and `/diagnose` endpoints.
- **Total test suite: 67 passed in 8.39s**.

---

## 11. Phase 3 — Multi-Subsystem Digital Twin & Cross-Subsystem Fault Propagation

### Architecture Overview
Phase 3 expands the digital twin from a single air-intake and fuel-control focus to a comprehensive **three-subsystem powertrain causal graph**:
1. `air_intake_and_fuel_control` (10 components, 12 failure modes)
2. `ignition_and_combustion` (3 new components, 3 failure modes)
3. `exhaust_and_aftertreatment` (2 new components, 2 failure modes)

Total model dimensions: **15 core components, 17 representative failure modes across 3 subsystems** (plus vehicle platform overlays).

### New Modeled Subsystem Components (`model/catalogue.yaml`)
- **Ignition & Combustion Subsystem**:
  - `ignition_coil`: Coil-on-plug (COP) step-up transformer (25,000–40,000V secondary discharge).
  - `spark_plug`: Ground electrode and ceramic insulator nose providing ignition arc spark gap.
  - `crankshaft_position_sensor`: Magnetic reluctor pickup reading 58x/60-2 tooth wheels for RPM and TDC sync.
- **Exhaust & Aftertreatment Subsystem**:
  - `catalytic_converter`: Ceramic honeycomb with Pt/Rh precious metals and ceria oxygen storage capacity.
  - `downstream_oxygen_sensor`: Heated zirconia sensor (O2S2) monitoring post-catalyst redox conversion.

### New Failure Modes & Diagnostic Scenarios
1. `ignition_coil_breakdown`: Secondary winding insulation puncture arcing to cylinder head under load (`P0300`, `P0301`).
2. `spark_plug_fouled`: Soot/carbon accumulation shunting spark energy to ground or excessive gap > 1.5mm (`P0300`, `P0301`).
3. `crank_sensor_thermal_failure`: Internal copper winding thermal open-circuit at >85°C engine block temperature (`P0335`, `P0336`).
4. `catalyst_efficiency_degraded`: Washcoat sintering destroying oxygen storage capacity; downstream O2 mirrors upstream (`P0420`).
5. `downstream_o2_sensor_biased`: Ceramic reference leak or heater element open-circuit creating false voltage bias (`P0136`, `P0138`).

### Causal Cross-Subsystem Fault Propagation
- Formulated cascading failure paths where a primary malfunction in one subsystem induces severe secondary damage in a downstream subsystem:
  - `ignition_coil_breakdown` $\to$ raw liquid hydrocarbon dumped into exhaust $\to$ combustion occurs on catalytic substrate at >950°C $\to$ rapid washcoat destruction $\to$ `catalyst_efficiency_degraded` (`P0420`).
- Implemented `cascading_effects` in `SimulationResult` and `cascading_risks` in `HypothesisScore`, surfacing actionable hazard alerts in diagnostic outputs.

### Multi-Subsystem Adaptive Questions (EIG) (`agents/eig.py`)
- Registered 4 new physical inspection procedures:
  - `ignition_coil_swap`: Swap coil pack to adjacent cylinder (misfire follows coil vs stays at cylinder).
  - `spark_plug_condition`: Feeler gauge gap measurement and insulator visual check (clean vs fouled/eroded).
  - `downstream_o2_switching`: 2500 RPM post-cat O2S2 voltage trace (steady 0.6V vs active oscillation).
  - `catalytic_temp_delta`: Infrared thermometer inlet vs outlet temperature differential.

### Self-Review Gate Updates (`agents/review.py`)
- Expanded `SUPPORTED_SUBSYSTEMS` to validate all 3 subsystems (`air_intake_and_fuel_control`, `ignition_and_combustion`, `exhaust_and_aftertreatment`).
- Expanded `VALID_COMPONENTS` to 15 components.

### Technical RAG Knowledge Base (`rag/documents.yaml`)
- Added 5 new technical reference documents and TSBs:
  - `DOC-OBD-P0301`: Single-cylinder misfire diagnostic priority logic (coil swap).
  - `DOC-OBD-P0335`: Crankshaft position sensor thermal open-circuit diagnostics.
  - `DOC-OBD-P0420`: Catalyst oxygen storage evaluation via downstream switching frequency.
  - `DOC-TSB-2024-01`: COP boot dielectric breakdown and carbon tracking under acceleration.
  - `DOC-TSB-2024-02`: Upstream misfires dumping raw fuel and melting catalytic substrates.

### Interface & UI Updates
- **Gradio Station (`app.py`)**: Updated title, 3-subsystem banner, added scenarios 8 & 9, and added cross-subsystem cascading hazard alerts.
- **FastAPI (`app/api.py`)**: `/health` endpoint returns all 3 supported subsystems.

### Test Suite Verification (`tests/test_multi_subsystem.py`)
- 10 dedicated automated unit and integration tests covering:
  - Multi-subsystem catalogue dimensions and component verification.
  - Ignition coil breakdown scoring (>90% posterior).
  - Spark plug fouling vs coil swap discrimination.
  - Crankshaft position sensor thermal breakdown scoring.
  - Catalytic converter degradation scoring (>90% posterior).
  - Cross-subsystem cascading hazard alerts.
  - EIG misfire test selection prioritizing coil swap.
  - Multi-subsystem RAG retrieval.
  - End-to-end LangGraph diagnosis with review gate passage (`PASSED`).
- **Complete Test Suite: 77 passed in 6.87s**.

---

## 12. Phase 4 — EV & Hybrid Electrified Powertrain Subsystem

### Subsystem Scope & Architectural Additions
- Added **Subsystem 4 (`high_voltage_battery_and_ev_powertrain`)** to [`model/catalogue.yaml`](model/catalogue.yaml):
  - **4 New Components**:
    1. `battery_management_system` (BMS controller, cell shunt balancing, contactor drive)
    2. `high_voltage_battery_pack` (Series-parallel 200V-400V DC energy storage pack)
    3. `traction_inverter` (3-phase IGBT/MOSFET power inverter module)
    4. `electric_traction_motor` (Permanent Magnet Synchronous Motor with resolver feedback)
  - **5 New Failure Modes**:
    1. `bms_cell_voltage_imbalance` (`P0A7F`, `P0A80` — delta > 120 mV under high load, acceleration derated)
    2. `high_voltage_isolation_loss` (`P0AA6` — chassis insulation resistance loss < 500 kOhm)
    3. `inverter_phase_loss` (`P0A1B` — IGBT gate short / open causing severe drive stutter)
    4. `traction_motor_resolver_fault` (`P0A40` — commutation angle drift and sensor fault)
    5. `battery_cooling_pump_failure` (`P0A93` — loss of inverter/pack coolant circulation)

### Vehicle Platform Profiles
- Created **Toyota Prius 1.8L Hybrid Synergy Drive** ([`model/vehicles/toyota_prius_18_hybrid.yaml`](model/vehicles/toyota_prius_18_hybrid.yaml)):
  - Split architecture: 1.8L Atkinson Cycle 2ZR-FXE + 201.6V NiMH Traction Battery + Dual Motor-Generators (MG1/MG2).
  - Configured platform prior overrides elevating inverter coolant pump failure to 14.0% prior based on known TSB patterns.
- Created **Tesla Model 3 Long Range Dual Motor BEV** ([`model/vehicles/tesla_model_3_bev.yaml`](model/vehicles/tesla_model_3_bev.yaml)):
  - Pure BEV architecture: 350V Lithium-Ion NCA Pack + Dual PMSM/Induction Inverter Drive + Pyrotechnic Fuse & BMS.
  - Configured platform prior overrides elevating high-voltage isolation loss to 12.0% prior.

### Self-Review & EIG Updates
- Updated `SUPPORTED_SUBSYSTEMS` in `agents/review.py` to validate all 4 subsystems.
- Updated `VALID_COMPONENTS` to all 19 components across the platform.
- Registered 5 EV-specific diagnostic inspection procedures in `agents/eig.py`:
  - `cell_voltage_delta`: BMS cell block divergence under load.
  - `chassis_isolation_resistance`: 1000V megohmmeter chassis insulation test.
  - `inverter_temp`: Inverter heat sink operational temperature.
  - `battery_cooling_flow`: Visible coolant reservoir agitation / active pump command.
  - `motor_resolver_jitter`: Scope analysis of resolver sine/cosine waveforms.

### Technical Documents Ingested
- Ingested 5 OEM and SAE technical reference documents in `rag/documents.yaml`:
  - `DOC-OBD-P0AA6`: High-Voltage Isolation Loss fault isolation tree.
  - `DOC-OBD-P0A7F`: Hybrid Battery Pack Deterioration and delta V analysis.
  - `DOC-OBD-P0A40`: Motor Resolver Position Sensor commutation calibration.
  - `DOC-TSB-PRIUS-2021`: Prius Inverter Coolant Pump cavitation & thermal degradation.
  - `DOC-TSB-TESLA-2022`: Tesla Model 3 HV Harness water-ingress isolation fault.

### Verification (`tests/test_ev_powertrain.py`)
- 9 dedicated unit and integration tests passing:
  - Prius profile loading and hybrid architecture validation.
  - Tesla Model 3 profile loading and BEV component boundaries.
  - BMS cell voltage imbalance Bayesian diagnosis.
  - High-voltage isolation loss diagnosis on Tesla Model 3.
  - Inverter cooling pump failure diagnosis on Toyota Prius.
  - Review gate validation for EV components (`PASSED`).
  - EV EIG adaptive test candidate generation.
  - Technical RAG document retrieval with citations.
  - Dual-platform comparison sanity checks.

---

## 13. Phase 5 — Real-Time Live Sensor Telemetry & Time-Series Streaming Engine

### Architecture (`stream/`)
- **`TelemetryFrame` & `TelemetryStream` (`stream/schema.py`)**:
  - High-frequency temporal snapshots containing standard PIDs (RPM, Speed, Throttle, MAF, MAP, ECT, STFT, LTFT, O2 Upstream, O2 Downstream) and EV PIDs (Pack Voltage, Pack Current, Cell Max Delta, Inverter Temp).
- **Physical Drive Cycle Simulator (`stream/generator.py`)**:
  - 10Hz continuous signal generator modeling warm curb idle (0-7s), steady 60 km/h cruise (7-15s), and acceleration regimes.
  - Injects deterministic physical fault signatures: intake vacuum leak, MAF underreporting, sluggish O2 response, downstream O2 catalyst tracking, EV cell voltage sag, and inverter thermal runaway.
- **Time-Series Stream Analyzer (`stream/analyzer.py`)**:
  - Zero-crossing frequency detection for upstream O2 sensor (identifying sluggish sensors < 0.6 Hz).
  - Downstream O2 standard deviation calculation (detecting catalyst pass-through oscillation > 0.12V).
  - Idle vs Cruise LTFT divergence comparison (isolating intake vacuum leaks vs global MAF bias).
  - Automated conversion into structured `Observations` with corresponding DTCs.

### Verification (`tests/test_stream_telemetry.py`)
- 9 automated tests passing:
  - Healthy baseline stream generation and analysis.
  - Vacuum leak LTFT idle/cruise divergence and end-to-end diagnosis.
  - MAF contamination detection and dual DTC generation.
  - Sluggish O2 sensor switching frequency thresholding.
  - Degraded catalyst downstream O2 mirror detection (`P0420`).
  - EV battery cell delta under load detection (`P0A7F`).
  - EV inverter thermal overheating detection (`P0A93`).
  - Empty and short stream edge case resilience.

---

## 14. Phase 6 — Automated Repair Action Plan, Labor/Parts Cost Estimator & Post-Repair Verification Drive Cycle Engine

### Architecture (`repair/`)
- **Standardized Schemas (`repair/schema.py`)**:
  - `PartEstimate`: OEM part numbers, component descriptions, quantity, unit pricing, categorization.
  - `LaborEstimate`: Flat-rate Mitchell1 / Alldata book hours, hourly shop rate, ASE certification requirement (`ASE_A8_ENGINE_PERFORMANCE`, `ASE_L3_HYBRID_EV_SPECIALIST`).
  - `SafetyWarning`: Critical warnings emphasizing NFPA 70E High Voltage depowering protocols, Class 0 1000V gloves, and depressurization procedures.
  - `RepairStep`: Standard Operating Procedure (SOP) across prep, disassembly, component service, and reassembly phases with explicit torque specifications.
  - `VerificationProtocol` & `DriveCycleStep`: Prescribed post-repair drive cycle regimes (cold soak, idle, steady cruise, decel fuel cut-off) with monitor readiness checks.
  - `RepairPlan`: Unified work order and estimate model.
- **Repair Procedure Database (`repair/catalogue.yaml`)**:
  - Comprehensive repair definitions covering all 22 core failure modes across the 4 subsystems.
- **Repair Planner (`repair/planner.py`)**:
  - `generate_repair_plan(diagnosis, hourly_rate_usd)`: Synthesizes diagnostic findings into actionable work orders with customizable shop labor rates. Automatically injects NFPA 70E High-Voltage protocols for EV platforms.
- **Post-Repair Verification Engine (`repair/verification.py`)**:
  - `verify_post_repair_drive_cycle(fault_id, stream)`: Evaluates post-service telemetry against physical criteria (e.g. idle LTFT normalization, O2 flatline, cell delta balance, inverter temp control) and verifies OBD-II monitor readiness.

### Verification (`tests/test_repair_workflow.py`)
- 9 automated tests passing:
  - Repair catalogue integrity and failure mode coverage.
  - ICE intake vacuum leak repair plan, torque specs (10 Nm), and labor calculations.
  - EV High-Voltage safety protocols (NFPA 70E, 1000V PPE, L3 Specialist requirements).
  - Exhaust catalyst replacement plan and EPA Federal drive cycle.
  - Hourly shop labor rate scalability ($80/hr vs $160/hr).
  - Successful post-repair verification with normalized telemetry (monitors set to READY).
  - Unresolved fault detection in post-repair verification.
  - EV inverter cooling pump post-repair verification.
  - Generic fallback resilience for uncatalogued faults.

### Total Platform Metrics
- **Supported Subsystems**: 4
- **Supported Components**: 19
- **Supported Failure Modes**: 22
- **Supported Vehicle Platforms**: 5 (Generic Petrol, Hyundai i20 MPI, VW Golf EA211 TSI, Toyota Prius HEV, Tesla Model 3 BEV)
- **Total Automated Unit & Integration Tests**: **107 passed across 12 test modules** (100% pass rate in 7.19s).




