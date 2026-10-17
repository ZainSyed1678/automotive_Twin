# AI-Powered Automotive Digital Twin for Fault Diagnosis

> **Simulation-Driven Fault Diagnosis for Vehicle Air-Intake & Fuel-Control Subsystems**  
> *A portfolio engineering project demonstrating transparent Bayesian likelihood scoring, digital twin fault propagation, and synthetic benchmark evaluation.*

---

## 1. Project Overview

Modern automotive diagnosis is fraught with ambiguity: a single generic OBD-II Diagnostic Trouble Code (DTC) such as `P0171` (System Too Lean) can stem from unmetered false air ingress, a failing fuel pump, a clogged fuel injector, or MAF sensor contamination.

This project implements an **AI-powered Automotive Digital Twin** that models component-level causal relationships and symptom propagation. Unlike black-box LLM systems that hallucinate confidence, this system:
1. **Evaluates diagnostic hypotheses against an explicit causal fault model** using transparent Log-Likelihood Ratio (Bayes Factor / Good's Weight of Evidence) scoring.
2. **Maintains strict evidence neutrality**: missing sensor observations are never penalized as negative evidence.
3. **Penalizes physical contradictions**: incompatible observations (e.g. normal fuel pressure when diagnosing a weak fuel pump) heavily discount conflicting hypotheses.
4. **Flags diagnostic ambiguity and insufficient evidence**: when leading candidates have small confidence gaps, the engine requests targeted physical tests instead of guessing.
5. **Evaluates against held-out synthetic benchmarks**: 200 stratified diagnostic cases across 5 difficulty levels with zero data leakage.

---

## 2. Subsystem Scope & Digital Twin Model

The initial implementation focuses on **one subsystem**:
**Engine air-intake and fuel-control subsystem of a generic 4-stroke petrol car.**

### Supported Components (10)
- **Mass Air Flow (MAF) Sensor**: Measures mass intake airflow for injection calculation.
- **Manifold Absolute Pressure (MAP) Sensor**: Measures intake manifold depression for load/density inference.
- **Throttle Position Sensor (TPS)**: Dual-potentiometer wiper monitoring driver throttle request.
- **Upstream Heated Oxygen Sensor (O2S1)**: Switching zirconia sensor measuring exhaust oxygen content.
- **Engine Coolant Temperature (ECT) Sensor**: Negative Temperature Coefficient (NTC) thermistor.
- **Fuel Injector Assembly**: Solenoid-actuated fuel metering valves.
- **Fuel Pump & Delivery Assembly**: Electric in-tank roller/vane pump and delivery lines.
- **Intake Manifold & Vacuum System**: Plenum, runner gaskets, and vacuum signal hoses.
- **Subsystem Wiring Harness & Connectors**: Signal conductors, 5V reference, chassis grounds, and terminals.
- **Engine Control Unit (ECU)**: Microcontroller executing fuel trim algorithms and OBD-II diagnostics.

### Modeled Fault Catalogue (12 Representative Failure Modes)
Each failure mode in [`model/catalogue.yaml`](model/catalogue.yaml) defines prior probability, conditional DTC likelihoods $P(\text{DTC} \mid F)$, conditional symptom state distributions $P(\text{symptom} \mid F)$, and physical distinguishing tests:

1. `maf_contamination` (MAF sensor hot-wire film deposit)
2. `intake_vacuum_leak` (Unmetered post-throttle air leak)
3. `map_sensor_drift` (Diaphragm fatigue / sensor offset)
4. `o2_sensor_slow_response` (Zirconia element silica/carbon aging)
5. `fuel_injector_clogged` (Single-cylinder basket/pintle varnish restriction)
6. `fuel_pump_weak` (Electric motor brush wear / flow volume degradation)
7. `connector_corrosion` (Pin fretting / terminal oxidation)
8. `wiring_open_circuit` (Broken conductor / high resistance)
9. `wiring_short_circuit` (Conductor shorted to ground or reference voltage)
10. `coolant_sensor_stuck_low` (ECT fixed at high electrical resistance / sub-zero)
11. `throttle_sensor_drift` (Resistive wiper track wear at low throttle angle)
12. `fuel_pressure_regulator_stuck_open` (Internal bypass spring fatigue / pressure bleed-down)

---

## 3. Mathematical Foundations: Transparent Likelihood Scoring

Rather than relying on opaque probabilistic graphs or black-box LLMs, diagnostic scoring uses **Good's Weight of Evidence (Turing-Bayes Factor)**:

$$\text{Log-Likelihood Score}(F) = \log P(F) + \sum_{o \in \text{Obs}} \log\left(\frac{P(o \mid F)}{P_{\text{baseline}}(o)}\right)$$

Where:
- **Prior Probability $P(F)$**: Domain frequency baseline for the failure mode ($0.05$ to $0.12$).
- **Baseline Ambient Frequencies $P_{\text{baseline}}(o)$**:
  - $P_{\text{baseline}}(\text{DTC}) = 0.02$ (DTCs are rare in healthy vehicles).
  - $P_{\text{baseline}}(\text{abnormal symptom}) = 0.08$ (e.g. rough idle, hissing noise, low pressure).
  - $P_{\text{baseline}}(\text{normal symptom}) = 0.85$ (e.g. normal pressure, clean pins).

### Core Diagnostic Rules
1. **Missing Evidence is Neutral**: If a technician has not yet measured fuel rail pressure, that feature is omitted from the sum ($W = 0.0$). Unperformed tests do not harm a hypothesis.
2. **Contradictory Evidence is Penalized**: If a fault requires low pressure ($P=0.95$), but the technician enters `fuel_pressure: normal` ($P=0.04$), the penalty $\log(0.04 / 0.85) = -3.05$ down-ranks the hypothesis immediately.
3. **Posterior Probability Normalization**: Raw log scores are converted to normalized posterior probabilities via numerically stable softmax:
   $$P(F \mid \text{Obs}) = \frac{\exp(\text{Score}(F) - \max_j \text{Score}_j)}{\sum_k \exp(\text{Score}_k - \max_j \text{Score}_j)}$$
4. **Ambiguity & Insufficient Evidence Detection**:
   - $\text{Confidence Gap} = P(F_{\text{top1}}) - P(F_{\text{top2}})$.
   - If $\text{Gap} < 0.15$: Flagged as **`is_ambiguous = True`**, and targeted distinguishing tests are recommended.
   - If no observations exist or $P(F_{\text{top1}}) < 0.25$: Flagged as **`insufficient_evidence = True`**.

---

## 4. Benchmark Dataset & Anti-Leakage Architecture

Milestone 2 introduced a reproducible synthetic case generator ([`casegen/`](casegen/)) producing 200 diagnostic cases:

### Dataset Splits
- **Development Set (`data/cases_dev.json`)**: 140 cases (70.0%)
- **Held-Out Evaluation Set (`data/cases_eval.json`)**: 60 cases (30.0%)
- **Target Difficulty Distribution (Exact 70% Dev / 30% Eval per tier)**:
  - **Straightforward (35%, 70 cases)**: Distinguishing DTCs and symptoms present.
  - **Incomplete / Sparse (25%, 50 cases)**: DTC dropout (30–60%) and pruned symptoms (1–2 observations only).
  - **Ambiguous (20%, 40 cases)**: Shared feature overlap (e.g. `P0171` without fuel rail pressure or smoke test data).
  - **Misleading / Noisy (10%, 20 cases)**: Injected secondary distractor symptom or uncharacteristic code.
  - **Insufficient Evidence (10%, 20 cases)**: Sparse non-distinguishing technician observations where the engine must abstain.

### Ground-Truth Isolation & Anti-Leakage
1. **Allowlist Agent Cases (`data/agent_cases/`)**: Agent-visible files are serialized using an immutable allowlist containing strictly `{"case_id", "observations"}`. Fields like `true_fault_id`, `difficulty`, and metadata are physically absent.
2. **Canonical Observation Fingerprinting**:
   $$\text{Fingerprint} = \text{SHA256}(\text{JSON}(\text{sorted\_dtcs}, \text{sorted\_symptoms}))$$
   - Invariant to DTC or symptom dictionary insertion order.
   - **Leakage check**: $\text{fingerprints}(\text{dev}) \cap \text{fingerprints}(\text{eval}) = \emptyset$ (Verified: **0 collisions**).

---

## 5. Repository Structure

```text
digital/
├── casegen/                   # M2: Synthetic Case Generator
│   ├── __init__.py
│   ├── schema.py              # GroundTruthCase vs AgentCase schemas
│   ├── generator.py           # Seeded case generator with noise injection
│   └── split.py               # Stratified splitting and fingerprint anti-leakage
├── data/
│   ├── cases_dev.json         # 140 dev cases (with ground-truth labels)
│   ├── cases_eval.json        # 60 held-out evaluation cases
│   ├── dataset_manifest.json  # Dataset metadata and verification metrics
│   └── agent_cases/           # Sanitized datasets for agent consumption
│       ├── cases_dev.json     # Stripped of true_fault_id and difficulty
│       └── cases_eval.json    # Stripped of true_fault_id and difficulty
├── model/                     # M1: Digital Twin Fault Model & Engine
│   ├── __init__.py
│   ├── schema.py              # Pydantic schemas (Component, Fault, Observations)
│   ├── catalogue.yaml         # Declarative causal graph and likelihood distributions
│   └── engine.py              # FaultModel class with simulate_fault() and score_hypotheses()
├── tests/
│   ├── test_model.py          # M1 unit and diagnostic scenario tests
│   └── test_casegen.py        # M2 schema, isolation, reproducibility, and leakage tests
├── generate_data.py           # CLI script to regenerate datasets
├── run_demo.py                # Interactive CLI demonstration
├── pytest.ini                 # Pytest configuration (pythonpath = .)
├── progress.md                # Persistent engineering log and architectural decisions
└── README.md                  # Project documentation and specifications
```

---

## 6. Quick Start & Verification

### Prerequisites
- Python 3.11+
- Dependencies: `pydantic`, `pyyaml`, `pytest`

### 1. Run the Diagnostic Demo
Demonstrates fault simulation, an ambiguous case, and disambiguated real-world cases:
```powershell
python run_demo.py
```

### 2. Run the Full Test Suite
Runs all 17 automated tests (M1 regression + M2 casegen):
```powershell
pytest -v
```

Expected output:
```text
tests/test_casegen.py::test_ground_truth_case_schema PASSED              [  5%]
tests/test_casegen.py::test_agent_case_forbids_extra_fields PASSED       [ 11%]
tests/test_casegen.py::test_agent_projection_and_allowlist_serialization PASSED [ 17%]
tests/test_casegen.py::test_exported_agent_files_contain_no_ground_truth PASSED [ 23%]
tests/test_casegen.py::test_generator_reproducibility PASSED             [ 29%]
tests/test_casegen.py::test_different_seeds_produce_different_datasets PASSED [ 35%]
tests/test_casegen.py::test_fingerprint_order_insensitivity PASSED       [ 41%]
tests/test_casegen.py::test_split_target_and_anti_leakage PASSED         [ 47%]
tests/test_casegen.py::test_straightforward_cases_feasibility PASSED     [ 52%]
tests/test_casegen.py::test_ambiguous_cases_feasibility PASSED           [ 58%]
tests/test_casegen.py::test_insufficient_evidence_cases_feasibility PASSED [ 64%]
tests/test_model.py::test_catalogue_structure_and_counts PASSED          [ 70%]
tests/test_model.py::test_simulate_fault PASSED                          [ 76%]
tests/test_model.py::test_missing_evidence_neutrality PASSED             [ 82%]
tests/test_model.py::test_contradictory_evidence_downranks PASSED        [ 88%]
tests/test_model.py::test_insufficient_evidence_flag PASSED              [ 94%]
tests/test_model.py::test_handwritten_diagnostic_scenarios PASSED        [100%]
====== 17 passed in 0.93s ======
```

### 3. Regenerate Datasets
```powershell
python generate_data.py
```

---

## 7. Project Roadmap

| Phase | Milestone | Scope | Status |
|---|---|---|---|
| **Phase 1** | **M1: Fault Model** | Causal catalogue, `simulate_fault`, likelihood scoring, scenario tests | **Complete** |
| | **M2: Case Generator** | 200 synthetic cases (140/60 split), anti-leakage checks, label isolation | **Complete** |
| | **M3: Knowledge Base & MCP** | Ingestion of synthetic repair manuals & TSBs, hybrid search, Python MCP server | **Complete** |
| | **M4: Diagnostic Agent** | LangGraph orchestration (intake, retrieve, simulate, score), LLM/RAG baselines | **Complete** |
| | **M5: Adaptive Questioning** | Expected Information Gain question selection, evidence critique, review step | Next |
| | **M6: Interface & Tracing** | FastAPI backend, Streamlit diagnostic UI, Docker containerization | Pending |
| | **M7: Evaluation & Audit** | Full held-out benchmark across 3 modes, calibration (ECE/Brier), honest audit | Pending |
| **Phase 2** | Multi-Vehicle Support | Vehicle-specific configuration layer (Hyundai i20, etc.) | Future |

