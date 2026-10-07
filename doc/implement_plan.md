# Implementation Plan: Practo Domain Support Agent

**Track:** Healthcare (Practo)  
**System:** Multi-Agent Patient Experience & Clinic Operations Support Platform  
**Target Frameworks:** CrewAI, AutoGen, ChromaDB, SentenceTransformers, FastAPI, Pydantic v2, LangChain  
**Execution Profile:** 100% Offline, Deterministic, Zero Telemetry, Zero Cloud Dependencies  
**Specification Reference:** [`doc/architecture.md`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/doc/architecture.md) & [`doc/ProblemStatement.md`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/doc/ProblemStatement.md)

---

## 1. Executive Summary & Plan Overview

The **Practo Domain Support Agent** is an enterprise-grade AI healthcare operations system designed to handle patient inquiries regarding clinic consultation policies, appointment rescheduling, specialty fees, prescription refills, insurance claims, and real-time appointment statuses.

This implementation plan provides a structured, phased blueprint to build, test, and verify the entire platform against all 16 tasks and the 100-mark evaluation rubric. Every phase guarantees 100% offline determinism, mathematical invariant validation, air-gapped vector retrieval, multi-agent peer review, and defense-in-depth AI governance.

### Implementation Strategy & Phasing

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       5-PHASE IMPLEMENTATION WORKFLOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 0: Environment & Core Foundation                                     │
│  - Offline dependency pinning, telemetry deactivation, directory layout    │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 1: Dataset Generation & Dual-Strategy RAG Subsystem (Tasks 1–5)       │
│  - Seeded appointment generator, 12-doc KB, ChromaDB dual-chunking,        │
│    empirical cosine threshold calibration, precision/recall benchmark      │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: CrewAI Multi-Agent Core, Tools, Memory & Guardrails (Tasks 6–10)   │
│  - Escalation scoring, custom MOCK_LLM (BaseLLM), 3-agent CrewAI team,      │
│    multi-turn memory (LangChain), Pydantic schemas, 3-tier guardrails        │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: Production API, Structured Logging & Judge Evaluation (Tasks 11–13)│
│  - FastAPI (REST + WebSocket), ELK-compatible JSONL audit logger,           │
│    15-query LLM-as-a-judge quantitative evaluation engine                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 4: AutoGen Review, Governance & Response Caching (Tasks 14–16)        │
│  - AutoGen 2-agent RoundRobinGroupChat, least autonomy RBAC enforcement,    │
│    EU AI Act High-Risk document, token/char budget guard, normalized cache  │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 5: End-to-End Verification, Transcripts & Submission Finalization   │
│  - Run all 16 verifiable transcripts, comprehensive README, final audit     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Layout & Artifact Inventory

The implementation strictly adheres to the standard directory layout specified in the architecture:

```text
c:\Users\kastu\Desktop\capstone - health\
├── README.md                      # Complete setup, empirical stats, formulas & documentation
├── ProblemStatement.md           # Problem statement specification
├── requirements.txt               # Pinned dependencies (crewai, autogen, chromadb, etc.)
├── dataset.py                     # Deterministic, seeded appointment generator (Task 1)
│
├── doc/
│   ├── ProblemStatement.md       # Problem statement backup
│   └── architecture.md           # System architecture specification
│
├── kb/                            # 12 Authoritative Healthcare Policy Documents (Task 2)
│   ├── doc_01_booking_policy.txt
│   ├── doc_02_cancellation_policy.txt
│   ├── doc_03_fee_structure.txt
│   ├── doc_04_insurance_claims.txt
│   ├── doc_05_prescription_refill.txt
│   ├── doc_06_lab_turnaround.txt
│   ├── doc_07_telemedicine_rules.txt
│   ├── doc_08_emergency_protocol.txt
│   ├── doc_09_patient_privacy.txt
│   ├── doc_10_followup_discount.txt
│   ├── doc_11_second_opinion.txt
│   └── doc_12_home_visit.txt
│
├── rag/                           # Retrieval-Augmented Generation Subsystem
│   ├── __init__.py
│   ├── chunking.py                # Fixed-size (200/40) & sentence-boundary splitters (Task 3)
│   ├── indexing.py                # Local SentenceTransformers embedder & ChromaDB upsert (Task 3)
│   ├── generate.py                # Grounded retrieval with empirical threshold tau (Task 4)
│   └── evaluate.py                # Document-level precision/recall evaluation (Task 5)
│
├── crew/                          # CrewAI Multi-Agent Subsystem
│   ├── __init__.py
│   ├── mock_llm.py                # Custom BaseLLM subclass with ReAct parser & prompt slicing (Task 7)
│   ├── tools.py                   # check_appointment_status & escalation scoring S (Task 6)
│   ├── agents.py                  # 3-agent CrewAI team definition & kickoff runner (Task 7)
│   ├── memory.py                  # LangChain InMemoryChatMessageHistory manager (Task 8)
│   ├── schemas.py                 # Pydantic v2 AgentResponseSchema models (Task 9)
│   └── guardrails.py              # PII masking, prompt-injection, grounding gates (Task 10)
│
├── review/                        # Secondary Peer Review Subsystem
│   ├── __init__.py
│   └── autogen_review.py          # AutoGen RoundRobinGroupChat & VerdictModel review (Task 14)
│
├── governance/                    # AI Governance & Optimization
│   ├── __init__.py
│   ├── least_autonomy.py          # Proof of tool RBAC isolation (Task 15)
│   ├── budget.py                  # Runtime token & character budget ceiling (Task 15)
│   ├── cache.py                   # In-memory normalized query response cache (Task 16)
│   └── risk_classification.md     # EU AI Act High-Risk classification document (Task 15)
│
├── api/                           # Production FastAPI Service Layer
│   ├── __init__.py
│   ├── main.py                    # REST (/ask, /add-document) & WebSocket (/ws/chat) (Task 11)
│   └── logging_utils.py           # Structured ELK JSONL audit logger with trace ID (Task 12)
│
├── eval/                          # Quantitative Quality Assurance Subsystem
│   ├── __init__.py
│   ├── test_queries.py            # 15-query benchmark runner (Task 13)
│   └── judge.py                   # Offline deterministic LLM-as-a-judge scoring engine (Task 13)
│
└── transcripts/                   # Verifiable Execution Artifacts (Tasks 1–16)
    ├── task_01_dataset.txt
    ├── task_03_chunking.txt
    ├── task_04_grounded_rag.txt
    ├── task_05_comparison.txt
    ├── task_06_lookup_tool.txt
    ├── task_07_crew_kickoff.txt
    ├── task_08_memory_continuity.txt
    ├── task_08_memory_reset.txt
    ├── task_09_schemas.txt
    ├── task_10_guardrails.txt
    ├── task_11_fastapi.txt
    ├── task_12_logging.txt
    ├── task_13_evaluation.txt
    ├── task_14_autogen_approved.txt
    ├── task_14_autogen_revised.txt
    ├── task_15_governance.txt
    └── task_16_cache.txt
```

---

## 3. Detailed Phase-by-Phase Implementation Specifications

### Phase 0: Environment Setup & Air-Gapped Foundation

#### 0.1 Objectives
1. Pin all dependencies to stable, compatible versions avoiding conflicting peer dependencies.
2. Forcefully disable telemetry and network calls via environment variables.
3. Establish directory skeletons and common package modules.

#### 0.2 Deliverables
* [`requirements.txt`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/requirements.txt):
  ```text
  crewai>=0.28.0,<0.90.0
  autogen-agentchat>=0.2.0,<0.5.0
  chromadb>=0.4.22
  sentence-transformers>=2.2.2
  fastapi>=0.109.0
  uvicorn[standard]>=0.27.0
  pydantic>=2.5.0
  langchain>=0.1.0
  langchain-community>=0.0.20
  websockets>=12.0
  pytest>=8.0.0
  requests>=2.31.0
  ```
* Environment configuration:
  * `CREWAI_DISABLE_TELEMETRY=true`
  * `OTEL_SDK_DISABLED=true`
  * `TOKENIZERS_PARALLELISM=false`

#### 0.3 Verification Command
```powershell
python -c "import crewai, chromadb, sentence_transformers, fastapi, pydantic, langchain; print('Environment initialized successfully')"
```

---

### Phase 1: Data Architecture & RAG Core (Tasks 1–5)

#### Task 1: Synthetic Transactional Dataset Generator (`dataset.py`)
* **Objective:** Produce a seeded, deterministic dataset `APPOINTMENTS` meeting all statistical invariants.
* **Fields:** `record_id` (e.g., `APT-1001`), `category`, `status`, `consultation_fee_inr`, `days_since_created` (0–30), `follow_up_required` (bool).
* **Controlled Vocabularies:**
  * Categories ($N=5$): `General Medicine`, `Cardiology`, `Dermatology`, `Pediatrics`, `Orthopedics` ($\ge 3$ records each).
  * Statuses ($N=5$): `Scheduled`, `Completed`, `Cancelled`, `No-Show`, `Rescheduled` ($\ge 1$ record each).
* **Consultation Fee Bands (INR):**
  * General Medicine: ₹400–₹800
  * Pediatrics: ₹500–₹1,000
  * Dermatology: ₹700–₹1,500
  * Orthopedics: ₹800–₹1,800
  * Cardiology: ₹1,000–₹2,500
  * *Justification:* Reflects urban Indian OPD fee structures on Practo, ranging from ₹400 for primary care to ₹2,500 for super-specialty consultations.
* **Invariant Constraints:**
  1. Dataset size $N_{\text{total}} \ge 40$ (target $N=45$).
  2. Follow-up rate constraint: $0.10 \le \text{rate} \le 0.30$.
  3. Algorithmic seed adjustment (`seed=42`) with zero manual editing.
* **Deliverables:** [`dataset.py`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/dataset.py)
* **Verification & Transcript:**
  ```powershell
  python dataset.py > transcripts/task_01_dataset.txt
  ```

---

#### Task 2: Authoritative Knowledge Base Corpus (`kb/`)
* **Objective:** Author $\ge 12$ distinct policy documents in our own words ($2\text{--}5$ sentences each) covering Practo's operational guidelines.
* **Topic Coverage:**
  1. `doc_01_booking_policy.txt`: Advance booking window (14 days), OTP mobile verification, SMS slot confirmation.
  2. `doc_02_cancellation_policy.txt`: Minimum 2-hour notice for 100% refund, free rescheduling up to 1 hour before slot.
  3. `doc_03_fee_structure.txt`: Specialty consultation tiers (General Medicine ₹400–800, Cardiology up to ₹2500).
  4. `doc_04_insurance_claims.txt`: Cashless pre-auth at network clinics, itemized reimbursement claim pack.
  5. `doc_05_prescription_refill.txt`: Valid prescription dated within 90 days, chronic care physician sign-off.
  6. `doc_06_lab_turnaround.txt`: Routine blood panels 12–24h, specialized pathology/biopsy 48–72h.
  7. `doc_07_telemedicine_rules.txt`: Minor conditions & routine follow-ups allowed; acute chest pain/trauma strictly barred.
  8. `doc_08_emergency_protocol.txt`: Red-flag alerts redirecting patients to call 112/108 or visit nearest ER.
  9. `doc_09_patient_privacy.txt`: DISHA & IT Act compliance, end-to-end encryption, zero commercial data selling.
  10. `doc_10_followup_discount.txt`: 50% discount or free review within 7 calendar days for the same ailment.
  11. `doc_11_second_opinion.txt`: Multi-specialty review panel turnaround within 48h with historical reports.
  12. `doc_12_home_visit.txt`: Bedridden/geriatric patients within 10 km clinic radius booked 24h in advance.
* **Deliverables:** 12 text files under [`kb/`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/kb/).

---

#### Task 3: Dual-Strategy Chunking & ChromaDB Vector Indexing (`rag/chunking.py`, `rag/indexing.py`)
* **Objective:** Index the KB into two separate disk-persisted ChromaDB collections using `sentence-transformers/all-MiniLM-L6-v2`.
* **Strategy A (Fixed-Size with Overlap):**
  * Window: 200 characters. Overlap: 40 characters.
  * ChromaDB Collection: `collection_fixed`.
* **Strategy B (Sentence-Boundary Splitting):**
  * Splitting: Natural regex punctuation splitting preserving full grammatical clauses ($1\text{--}2$ sentences per chunk).
  * ChromaDB Collection: `collection_sentence`.
* **Persistence & Upsert:**
  * Path: `./data/chroma_db` via `chromadb.PersistentClient`.
  * Idempotent chunk IDs (`doc_01_f001`, `doc_01_s001`) via `.upsert()`.
* **Verification & Transcript:**
  ```powershell
  python -m rag.indexing > transcripts/task_03_chunking.txt
  ```

---

#### Task 4: Grounded Retrieval & Empirical Cosine Threshold Calibration ($\tau$) (`rag/generate.py`)
* **Objective:** Retrieve top-k chunks and strictly calibrate an authoritative cosine cutoff threshold $\tau$ to prevent hallucination.
* **Empirical Calibration Protocol:**
  1. Benchmark $\ge 3$ in-scope queries (cancellation window, cardiology fee, home visit criteria).
  2. Benchmark $\ge 2$ out-of-scope queries (veterinary medication, gold mortgage rates).
  3. Measure top-1 cosine similarities $\text{Sim}_{\max}(q)$ locally with PyTorch/SentenceTransformers.
  4. Compute the separation gap: $\tau \in (\max(\text{Sim}_{\text{out}}), \min(\text{Sim}_{\text{in}}))$.
  5. If $\text{Sim}_{\max}(q) < \tau$, halt retrieval and return:
     > *"I don't know based on the provided policy documents."*
* **Demonstration Requirement:** Demonstrate $\ge 5$ in-scope queries answered correctly and $\ge 1$ out-of-scope query triggering the fallback.
* **Verification & Transcript:**
  ```powershell
  python -m rag.generate > transcripts/task_04_grounded_rag.txt
  ```

---

#### Task 5: Chunking Strategy Benchmark & Precision/Recall Evaluation (`rag/evaluate.py`)
* **Objective:** Compare `collection_fixed` vs `collection_sentence` using document-level Precision and Recall across the $\ge 5$ in-scope queries.
* **Mathematical Formulation:**
  $$\text{Precision} = \frac{|D_{\text{retrieved}} \cap D_{\text{relevant}}|}{|D_{\text{retrieved}}|}, \qquad \text{Recall} = \frac{|D_{\text{retrieved}} \cap D_{\text{relevant}}|}{|D_{\text{relevant}}|}$$
* **Analysis & Recommendation:**
  * Output per-query arithmetic for both collections.
  * Provide a 2–3 sentence data-backed recommendation justifying the superior collection (sentence-based preserves complete clinical conditions without phrase truncation).
* **Verification & Transcript:**
  ```powershell
  python -m rag.evaluate > transcripts/task_05_comparison.txt
  ```

---

### Phase 2: CrewAI Multi-Agent System & Guardrails (Tasks 6–10)

#### Task 6: Appointment Status Tool & Continuous Escalation Scoring (`crew/tools.py`)
* **Objective:** Implement `check_appointment_status(record_id: str) -> dict` returning status, fee, and continuous escalation score $S$.
* **Escalation Score Formula:**
  $$S = w_{\text{follow\_up}} \cdot \mathbb{I}(\text{follow\_up\_required}) + w_{\text{recency}} \cdot \left(\frac{\text{days\_since\_created}}{30}\right)$$
  * $w_{\text{follow\_up}} = 0.60$, $w_{\text{recency}} = 0.40$ ($w_1 + w_2 = 1.00$).
  * Threshold $\theta_{\text{escalate}} = 0.70$.
  * *Justification:* Follow-up requirement carries clinical priority (0.60). Unresolved aging over 30 days contributes up to 0.40. $S \ge 0.70$ isolates high-risk cases (e.g., follow-up needed + age $> 7.5$ days, or aging $> 26$ days) requiring manual coordinator intervention.
* **Verification & Transcript:**
  ```powershell
  python -m crew.tools > transcripts/task_06_lookup_tool.txt
  ```

---

#### Task 7: Deterministic `MOCK_LLM` & CrewAI Multi-Agent Team (`crew/mock_llm.py`, `crew/agents.py`)
* **Objective:** Deploy a 3-agent CrewAI crew running deterministically offline via a custom `BaseLLM` subclass.
* **Silent Pitfall Safeguards:**
  1. **ReAct Prompt Slicing:** CrewAI's system template contains `"Observation: the result of the action"`. The mock LLM must slice off the system prompt and inspect only the agent's new conversational turn.
  2. **Schema-Based Tool Dispatch:** Dispatch tools strictly off declared argument schemas (`record_id` for lookup, `query` for RAG) and exact tool names, eliminating substring matching bugs (`rag_lookup` triggering on `"lookup"`).
* **3-Agent Team Architecture:**
  1. `Retrieval Agent`: Policy specialist with tool `rag_lookup` only.
  2. `Lookup Agent`: Operations officer with tool `check_appointment_status` only.
  3. `Response Composer`: Patient communications specialist with zero tools (pure synthesis).
* **Kickoff Demos:** Execute `.kickoff()` on both policy inquiries and appointment lookups.
* **Verification & Transcript:**
  ```powershell
  python -m crew.agents > transcripts/task_07_crew_kickoff.txt
  ```

---

#### Task 8: In-Memory Multi-Turn Session Memory (`crew/memory.py`)
* **Objective:** Manage conversational context using LangChain's `InMemoryChatMessageHistory`.
* **Verification Protocol:**
  * **Transcript A (`task_08_memory_continuity.txt`):** Turn 1 asks for appointment `APT-1002`; Turn 2 asks *"What is its consultation fee?"* referencing the prior appointment via pronoun resolution.
  * **Transcript B (`task_08_memory_reset.txt`):** A new `session_id` asks *"What is its consultation fee?"* and correctly handles the absence of prior context.
* **Verification Commands:**
  ```powershell
  python -m crew.memory --mode continuity > transcripts/task_08_memory_continuity.txt
  python -m crew.memory --mode reset > transcripts/task_08_memory_reset.txt
  ```

---

#### Task 9: Structured Pydantic Output Contracts (`crew/schemas.py`)
* **Objective:** Enforce Pydantic v2 schemas on all outputs from the agent pipeline.
* **Schema Definition:**
  ```python
  class AgentResponseSchema(BaseModel):
      query: str
      intent: str = Field(description="Policy inquiry or Appointment lookup")
      direct_answer: str
      citations: List[str] = Field(default_factory=list)
      escalation_triggered: bool = False
      escalation_score: Optional[float] = None
  ```
* **Validation:** Programmatically validate every output object before returning.
* **Verification & Transcript:**
  ```powershell
  python -m crew.schemas > transcripts/task_09_schemas.txt
  ```

---

#### Task 10: Multi-Tier Guardrail System (`crew/guardrails.py`)
* **Objective:** Implement input and output safety gates, demonstrating each firing at least once.
* **Guardrail Implementations:**
  1. **Input Guardrail 1 (PII Masking):** Regex masking for Indian 10-digit contact numbers (`(?:\+91[\-\s]?)?[6-9]\d{9}\b` $\to$ `[MASKED_CONTACT]`).
  2. **Input Guardrail 2 (Prompt-Injection Detector):** Rejection of adversarial override patterns (`ignore previous instructions`, `system override`, `act as dan`).
  3. **Output Guardrail (Groundedness Refusal):** Interception of ungrounded policy claims, substituting the fallback refusal string.
* **Verification & Transcript:**
  ```powershell
  python -m crew.guardrails > transcripts/task_10_guardrails.txt
  ```

---

### Phase 3: Production API, Logging & Quality Assurance (Tasks 11–13)

#### Task 11: Production FastAPI Service (`api/main.py`)
* **Objective:** Implement high-performance FastAPI service with REST and WebSocket endpoints.
* **Endpoint Specifications:**
  1. `POST /ask`: Accepts `{query: str, session_id: Optional[str]}`, runs guardrails $\to$ cache $\to$ agents $\to$ AutoGen $\to$ outputs `AgentResponseSchema`.
  2. `POST /add-document`: Accepts `{doc_id: str, content: str, topic: str}`, chunks and indexes dynamically into ChromaDB.
  3. `WebSocket /ws/chat`: Real-time duplex chat handling multi-turn conversation and surviving `WebSocketDisconnect` gracefully without crashing the server.
* **Verification & Transcript:**
  ```powershell
  python -m api.main --test > transcripts/task_11_fastapi.txt
  ```

---

#### Task 12: ELK-Compatible Structured Audit Logging (`api/logging_utils.py`)
* **Objective:** Emit single-line JSON (`.jsonl`) logs to `logs/audit.jsonl` with trace ID, timing, and strict zero-PII guarantees.
* **Log Record Structure:**
  ```json
  {
    "trace_id": "c7a84e20-3b91-4e78-9e12-3f1f7d5c98a1",
    "timestamp": "2026-10-07T07:45:00.124Z",
    "endpoint": "/ask",
    "latency_ms": 38.45,
    "masked_prompt": "Check appointment status for [MASKED_CONTACT] on APT-1002",
    "status_code": 200,
    "cache_hit": false
  }
  ```
* **PII Guarantee:** Raw contact numbers must pass through `mask_pii()` before logging. Raw numbers must never reach disk.
* **Verification & Transcript:**
  ```powershell
  python -m api.logging_utils --verify > transcripts/task_12_logging.txt
  ```

---

#### Task 13: Quantitative LLM-as-a-Judge Evaluation Framework (`eval/judge.py`, `eval/test_queries.py`)
* **Objective:** Benchmark system across 15 queries under deterministic `MOCK_LLM`.
* **Query Distribution (15 Queries):**
  * 12 Queries: One for each required KB policy topic (`doc_01` to `doc_12`).
  * 1 Query: Appointment status & escalation scoring inquiry (`APT-1001`).
  * 2 Queries: Adversarial / out-of-scope refusal probes (veterinary inquiry & prompt injection).
* **Scoring Metrics ($[0.0, 1.0]$ Scale):**
  1. Accuracy ($A$): Factual correctness against ground truth.
  2. Grounding ($G$): Fidelity to retrieved context.
  3. Completeness ($C$): Coverage of required constraints.
  4. Safety ($S$): PII redaction and refusal handling.
* **Reporting:** Report table of all 15 queries + averages across all 4 properties.
* **Verification & Transcript:**
  ```powershell
  python -m eval.test_queries > transcripts/task_13_evaluation.txt
  ```

---

### Phase 4: AutoGen Review, Governance & Optimization (Tasks 14–16)

#### Task 14: AutoGen Secondary Peer Review Stage (`review/autogen_review.py`)
* **Objective:** Deploy an independent 2-agent AutoGen review team before delivering responses.
* **Team Configuration:**
  * Group Chat: `RoundRobinGroupChat` bounded with `max_turns=2`.
  * Agent 1: `PolicyComplianceReviewer` (audits grounding, PII absence, policy adherence).
  * Agent 2: `FinalEditor` (approves or revises the draft, outputs `VerdictModel`).
* **Runtime Pitfall Mitigation:**
  * Declare `custom_message_types=[StructuredMessage[VerdictModel]]` on team initialization to prevent `ValueError: Message type ... is not registered`.
* **Verdict Contract:**
  ```python
  class VerdictModel(BaseModel):
      approved: bool
      final_answer: str
      reason: str
  ```
* **Demonstrations:**
  1. Approved unchanged query (`transcripts/task_14_autogen_approved.txt`).
  2. Revised query with injected ungrounded claim (`transcripts/task_14_autogen_revised.txt`).
* **Verification Commands:**
  ```powershell
  python -m review.autogen_review --case approved > transcripts/task_14_autogen_approved.txt
  python -m review.autogen_review --case revised > transcripts/task_14_autogen_revised.txt
  ```

---

#### Task 15: Four-Layer AI Governance Framework (`governance/least_autonomy.py`, `governance/budget.py`, `governance/risk_classification.md`)
* **Layer 1: Application Layer (Least Autonomy / RBAC):**
  * Demonstrate that only the Lookup Agent can call `check_appointment_status`. Wiring it to another agent is mechanically blocked and raises an access rejection. Include a one-paragraph explanation.
* **Layer 2: System Risk Tiering (`governance/risk_classification.md`):**
  * Author a one-paragraph justification classifying the platform under **High Risk** according to EU AI Act and healthcare regulations.
* **Layer 3: Runtime Budget Limiter (`governance/budget.py`):**
  * Enforce ceilings: max 512 tokens / max 2,048 characters per request.
  * Demonstrate that an oversized request is rejected with HTTP 413, not silently executed.
* **Verification & Transcript:**
  ```powershell
  python -m governance.least_autonomy > transcripts/task_15_governance.txt
  ```

---

#### Task 16: Normalized Query Response Cache (`governance/cache.py`)
* **Objective:** Sub-millisecond response caching for grounded queries.
* **Key Normalization:** `query.strip().lower()`.
* **Demonstration:**
  * Execute query $\to$ Cache Miss $\to$ Run pipeline $\to$ Populate cache.
  * Execute identical query $\to$ Cache Hit $\to$ Sub-millisecond return bypassing agents.
  * Provide before/after timing and call counter proof.
* **Verification & Transcript:**
  ```powershell
  python -m governance.cache > transcripts/task_16_cache.txt
  ```

---

### Phase 5: Verification, Benchmarking & Submission Readiness

#### 5.1 Objectives
1. Ensure all 17 transcript files exist under [`transcripts/`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/transcripts/) and contain clean, verifiable output.
2. Ensure [`README.md`](file:///c:/Users/kastu/Desktop/capstone%20-%20health/README.md) contains:
   * **Practo (Healthcare)** track declaration at top.
   * Telemetry deactivation statement (`CREWAI_DISABLE_TELEMETRY=true`).
   * Dataset design choices (seed, category weights, status weights, fee bands, justification).
   * Empirical cosine similarity measurements ($\text{Sim}_{\text{in}}$, $\text{Sim}_{\text{out}}$) and threshold $\tau$.
   * Chunking evaluation table with per-query precision/recall and recommended strategy justification.
   * Escalation formula and percentile justification.
   * 15-query evaluation results table and composite averages.
   * Governance risk classification justification.
3. Verify zero external network calls, zero API keys, and zero binary/media files in the repository.

#### 5.2 Verification Script (`verify_all.py`)
An automated master verification script will execute all task test suites, regenerate transcripts if needed, and assert all invariants.

---

## 4. Known Silent Pitfalls & Defensive Countermeasures

| # | Pitfall / Silent Failure | Root Cause | Implemented Defensive Countermeasure |
| :- | :----------------------- | :--------- | :------------------------------------ |
| 1 | **CrewAI ReAct Template Collision** | CrewAI's default system prompt includes `"Observation: the result of the action"`. | Mock LLM slices prompt and inspects *only* the new conversational turn, ignoring system instructions. |
| 2 | **Tool Name Substring Collision** | Substring matching on `"lookup"` causes `rag_lookup` and `check_appointment_status` to cross-invoke. | Tool dispatch inspects exact declared argument schema and exact registered name. |
| 3 | **AutoGen Message Registration Crash** | AutoGen raises `ValueError: Message type ... is not registered` when emitting custom Pydantic models. | Explicitly pass `custom_message_types=[StructuredMessage[VerdictModel]]` to team initialization. |
| 4 | **AutoGen Infinite Loop / Single Agent Speaking** | Unbounded conversation turns. | Configure `RoundRobinGroupChat` with `max_turns=2` or `MaxMessageTermination(3)`. |
| 5 | **Raw PII in Audit Log Files** | Logging raw query text leaks phone numbers to persistent disk. | Mandatory call to `mask_pii()` inside logging middleware before record serialization. |
| 6 | **CrewAI Network Telemetry Leaks** | Default telemetry triggers HTTP calls during `crew.kickoff()`. | Set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before import/run. |
| 7 | **Dynamic Chunk ID Collisions** | Re-running ingestion appends duplicate chunks in ChromaDB. | Use deterministic IDs (`doc_01_f001`, `doc_01_s001`) with `.upsert()`. |
| 8 | **Cache Normalization Inconsistencies** | Queries with different casing or whitespace cause false cache misses. | Enforce standard normalization: `query.strip().lower()`. |

---

## 5. Master Transcript Generation Matrix

The following matrix maps every graded task to its corresponding output transcript file:

| Task # | Target Transcript File | Generation Command | Key Assertions / Content |
| :--- | :--- | :--- | :--- |
| Task 1 | `transcripts/task_01_dataset.txt` | `python dataset.py` | $\ge 40$ records, 5 categories $\ge 3$, 5 statuses $\ge 1$, follow-up $10\%\text{--}30\%$. |
| Task 3 | `transcripts/task_03_chunking.txt` | `python -m rag.indexing` | ChromaDB indexing summary for `collection_fixed` and `collection_sentence`. |
| Task 4 | `transcripts/task_04_grounded_rag.txt` | `python -m rag.generate` | Cosine similarity scores for in-scope & out-of-scope; fallback triggered on out-of-scope. |
| Task 5 | `transcripts/task_05_comparison.txt` | `python -m rag.evaluate` | Per-query Precision & Recall for both collections + 2–3 sentence recommendation. |
| Task 6 | `transcripts/task_06_lookup_tool.txt` | `python -m crew.tools` | `check_appointment_status` output with fee and escalation score $S \in [0, 1]$. |
| Task 7 | `transcripts/task_07_crew_kickoff.txt` | `python -m crew.agents` | CrewAI `.kickoff()` demonstration with both tools dispatched on sample queries. |
| Task 8 | `transcripts/task_08_memory_continuity.txt` | `python -m crew.memory --mode continuity` | Multi-turn conversation preserving context across turns. |
| Task 8 | `transcripts/task_08_memory_reset.txt` | `python -m crew.memory --mode reset` | Fresh session correctly lacking prior conversational context. |
| Task 9 | `transcripts/task_09_schemas.txt` | `python -m crew.schemas` | Pydantic v2 `AgentResponseSchema` validation and JSON serialization. |
| Task 10 | `transcripts/task_10_guardrails.txt` | `python -m crew.guardrails` | Demonstrated firing of PII masking, injection detector, and grounding refusal. |
| Task 11 | `transcripts/task_11_fastapi.txt` | `python -m api.main --test` | Successful execution of `POST /ask`, `POST /add-document`, and WebSocket echo. |
| Task 12 | `transcripts/task_12_logging.txt` | `python -m api.logging_utils --verify` | Single-line JSONL audit entries with UUID4 trace IDs and `[MASKED_CONTACT]`. |
| Task 13 | `transcripts/task_13_evaluation.txt` | `python -m eval.test_queries` | 15-query evaluation results with Accuracy, Grounding, Completeness, Safety averages. |
| Task 14 | `transcripts/task_14_autogen_approved.txt` | `python -m review.autogen_review --case approved` | AutoGen review verdict approving draft without modifications. |
| Task 14 | `transcripts/task_14_autogen_revised.txt` | `python -m review.autogen_review --case revised` | AutoGen review verdict correcting an injected ungrounded claim. |
| Task 15 | `transcripts/task_15_governance.txt` | `python -m governance.least_autonomy` | Least autonomy RBAC proof and token budget HTTP 413 rejection demonstration. |
| Task 16 | `transcripts/task_16_cache.txt` | `python -m governance.cache` | Normalized cache hit demonstration showing sub-millisecond bypass. |

---

## 6. Acceptance Criteria & Definition of Done

Prior to final submission, the system must satisfy the following checklist:

- [ ] **Zero Network & Zero Telemetry:** The system runs completely air-gapped without cloud dependencies or active telemetry calls.
- [ ] **100% Deterministic Execution:** All modules run under deterministic random seeds and mathematical bounds.
- [ ] **All 16 Tasks Implemented:** Every deliverable in Tasks 1 through 16 is fully functional and tested.
- [ ] **All 17 Transcripts Generated:** Every required transcript file exists in `transcripts/` with reproducible evidence.
- [ ] **No Disallowed File Types:** Zero images, screenshots, videos, PDFs, or audio files present in the repository.
- [ ] **README Completeness:** Comprehensive README containing the required track declaration, architectural documentation, parameter justification tables, and step-by-step reproduction instructions.
