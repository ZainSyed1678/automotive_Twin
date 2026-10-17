# AI-Powered Automotive Digital Twin for Fault Diagnosis — Project Progress

## 1. Project Overview & Architecture Status
- **Goal**: Portfolio-grade AI-powered digital twin for simulation-driven vehicle fault diagnosis.
- **Domain**: Air-intake and fuel-control subsystem of a generic petrol vehicle.
- **Methodology**: Transparent Bayesian likelihood scoring (Good's weight of evidence / Bayes factor), simulation-driven hypothesis testing, synthetic case evaluation, and RAG knowledge retrieval.
- **Guiding Principles**: Ponytail engineering (YAGNI, minimal dependencies, stdlib-first, transparent math over black boxes, rigorous test coverage).

---

## 2. Milestone Roadmap & Progress

| Milestone | Title | Status | Details |
|---|---|---|---|
| **M1** | **Fault Model & Scoring Engine** | **Completed** | 10 components, 12 faults, causal catalogue, log-likelihood ratio scoring, 6/6 tests passing, demo runner verified. |
| **M2** | **Diagnostic Case Generator** | **Completed** | 200 synthetic cases (140 dev, 60 eval), 5 difficulty tiers, 0 fingerprint collisions, allowlist agent isolation, 17/17 tests passing. |
| **M3** | **Knowledge Base & MCP Tools** | Pending | Repair manuals, TSBs, generic OBD-II info, vector/hybrid retrieval, read-only MCP server. |
| **M4** | **Diagnostic Agent Core** | Pending | LangGraph orchestration (intake, retrieve, hypothesize, score, report), LLM-only & RAG-only baselines. |
| **M5** | **Adaptive Questioning & Review** | Pending | Information-gain question selection, evidence critique, insufficient-evidence handling. |
| **M6** | **Interface & Observability** | Pending | FastAPI backend, Streamlit diagnostic UI, Docker containerization. |
| **M7** | **Evaluation & Documentation** | Pending | Held-out benchmarking across 3 modes, calibration scoring, README accuracy audit. |

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
- **Documented Limitation**: An empty fingerprint intersection proves that zero exact duplicate observation sets exist across the split; it does not claim complete statistical independence of synthetic draws generated from the same underlying causal model.

### Test Verification (`tests/test_casegen.py`)
- 17 total tests passing (11 M2 tests + 6 M1 regression tests) in 0.93s:
  - `test_ground_truth_case_schema`: Valid GT cases accepted; missing required fields rejected.
  - `test_agent_case_forbids_extra_fields`: Extra fields on `AgentCase` forbidden.
  - `test_agent_projection_and_allowlist_serialization`: In-memory projection and allowlist dictionary verified.
  - `test_exported_agent_files_contain_no_ground_truth`: On-disk agent files verified for zero ground-truth keys.
  - `test_generator_reproducibility`: Bitwise identical generation under seed 42.
  - `test_different_seeds_produce_different_datasets`: Distinct seeds produce different cases.
  - `test_fingerprint_order_insensitivity`: Invariant to DTC/symptom ordering.
  - `test_split_target_and_anti_leakage`: 140/60 target and 0 fingerprint collision verified.
  - `test_straightforward_cases_feasibility`: Straightforward cases achieve $\ge 85\%$ Top-1 accuracy under M1 model.
  - `test_ambiguous_cases_feasibility`: Ambiguous cases demonstrate low confidence gaps and flag ambiguity.
  - `test_insufficient_evidence_cases_feasibility`: Insufficient-evidence cases do not falsely reveal confident diagnosis.
