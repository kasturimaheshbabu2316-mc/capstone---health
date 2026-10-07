# Problem Statement — Practo Domain Support Agent (CrewAI)

**Track:** Healthcare (Practo) · **Total marks:** 100 · **Duration:** 14 days
**Deliverable:** ONE public GitHub repository (dataset, RAG core, CrewAI crew, Autogen review stage, FastAPI deployment, transcripts, README)

---

## 1. Background

Practo's patient-experience team wants an AI support agent that gives patients instant, accurate answers about scheduling, fees, and follow-ups. The agent must:

1. Answer **clinic-policy questions** from a knowledge base we write ourselves (RAG).
2. **Look up a specific appointment's status** from a dataset we design and validate ourselves.
3. **Remember** the conversation within a session.
4. Be **guarded** against misuse (PII leakage, prompt injection, ungrounded answers).
5. Have its answers **reviewed by a second, independent agent team** (Autogen) before reaching the user.
6. Operate under an explicit **governance policy** (least autonomy, risk classification, cost budget).
7. Be **deployed** behind FastAPI (HTTP + WebSocket), **logged** in structured form, and **evaluated** end to end.

The goal is a system that could pass a production governance review, not just a happy-path demo.

---

## 2. Objectives

| # | Objective |
| --- | ----------- |
| O1 | Build a seeded, reproducible appointment dataset that meets strict statistical constraints |
| O2 | Author a ≥12-document knowledge base and index it with two chunking strategies in two ChromaDB collections |
| O3 | Implement grounded generation with an **empirically calibrated** "I don't know" threshold |
| O4 | Compare chunking strategies using document-level precision/recall and recommend one |
| O5 | Orchestrate a ≥3-agent CrewAI crew with tools, session memory, structured output, and guardrails |
| O6 | Deploy via FastAPI with HTTP endpoints plus a disconnect-safe WebSocket |
| O7 | Log every request as JSON Lines with trace ID and timing, without leaking raw PII |
| O8 | Evaluate Accuracy, Grounding, Completeness, Safety across 15 queries via LLM-as-judge |
| O9 | Add an Autogen `RoundRobinGroupChat` review stage with structured verdicts |
| O10 | Apply four-layer AI governance: least autonomy, risk level, budget cap; add response caching |

---

## 3. Hard Constraints

- **Everything must run under `MOCK_LLM`**: deterministic, zero API keys, zero network access at runtime. A real LLM may be added behind an env flag, but is optional and never used in graded transcripts.
- **Free and local only**: SentenceTransformers (embeddings) + ChromaDB (vector store). No paid accounts or credit cards.
- **No images, screenshots, PDFs, slides, video, or audio.** Every deliverable is code or text in the repo.
- **Disable CrewAI telemetry**: set `CREWAI_DISABLE_TELEMETRY=true` (or `OTEL_SDK_DISABLED=true`) before `crew.kickoff()`, and say so in the README.
- **Fabricated data only**: never use real patient names, diagnoses, or insurance IDs.
- **Originality**: dataset, KB, code, and analysis must be our own work.
- **Single submission link** by the end of the 14-day window.

---

## 4. Scenario Vocabulary

**Categories** (use each ≥ once; may add more): `General Medicine`, `Cardiology`, `Dermatology`, `Pediatrics`, `Orthopedics`

**Statuses** (use each ≥ once; may add more): `Scheduled`, `Completed`, `Cancelled`, `No-Show`, `Rescheduled`

**Required knowledge-base topics (12):**

1. Appointment-booking policy
2. Cancellation / rescheduling window
3. Consultation-fee structure by specialty
4. Insurance-claim process
5. Prescription-refill policy
6. Lab-test turnaround times
7. Telemedicine eligibility
8. Emergency-visit protocol
9. Patient-data privacy policy
10. Follow-up-visit discount policy
11. Second-opinion process
12. Home-visit eligibility

**PII fields:**

| Field | Format | Handling |
| ------- | -------- | ---------- |
| Contact number | Fixed format | **Must be masked** by input guardrail and in logs |
| Patient name | Free text | Out of scope for masking; fabricated examples only |
| Diagnosis / condition | Free text | Out of scope for masking; fabricated examples only |
| Insurance ID | No universal format | Out of scope for masking; fabricated examples only |

---

## 5. Part 1 — Dataset Design & RAG Core (30 marks)

### Task 1 — Appointment dataset (`dataset.py`)

- Seeded, deterministic generator producing `APPOINTMENTS` with **≥40 records**.
- Fields per record: `record_id`, `category`, `status`, `consultation_fee_inr`, `days_since_created` (int, 0–30), `follow_up_required` (bool).
- Constraints: every category ≥3 records; every status ≥1 record; `follow_up_required=True` between **10% and 30%**.
- If the band is missed, **change seed or weights and regenerate. Never hand-edit records.**
- Print and report: count per category, count per status, follow-up percentage.
- State a one-sentence justification for the chosen fee range (INR).
- README must record the exact **seed, category weights, status weights, amount range**.

### Task 2 — Knowledge base

- ≥12 documents, 2–5 sentences each, in our own words, covering all 12 required topics.

### Task 3 — Two chunking strategies, two collections

- **Fixed-size with overlap** → its own ChromaDB collection.
- **Sentence-based** → its own ChromaDB collection.
- Embed with a local SentenceTransformers model; index via `collection.upsert()`.

### Task 4 — Grounded generation

- Retrieve top-k chunks, answer using **only** retrieved context.
- **Calibrate the threshold empirically**: measure top-1 cosine similarity for ≥3 in-scope and ≥2 out-of-scope queries, then place the threshold between the two observed clusters. Do not use untested presets (0.5 / 0.6 / 0.7).
- Report measured values and the chosen threshold in the README.
- Demonstrate ≥5 in-scope queries and ≥1 out-of-scope query that triggers the fallback.

### Task 5 — Compare chunking strategies

- Same ≥5 queries as Task 4.
- Compute **document-level** precision and recall (map chunks → parent doc, dedup before scoring), **separately per collection**, with per-query arithmetic shown.
- Write 2–3 sentences recommending a strategy, citing both sets of numbers.

---

## 6. Part 2 — CrewAI Orchestration, Tools, Memory & Guardrails (30 marks)

Uses Part 1's RAG (recommended collection) as a fixed input.

### Task 6 — `check_appointment_status(record_id: str) -> dict`

- Returns `status`, `consultation_fee_inr`, and a **designed** `escalation_score ∈ [0, 1]`.
- Score combines `follow_up_required` with a normalized recency signal from `days_since_created` (not a bare boolean OR).
- State the **exact formula** and an escalation threshold justified from the dataset's own distribution (e.g., 80th percentile of `days_since_created`).

### Task 7 — CrewAI crew (≥3 agents)

- **Retrieval Agent** (RAG tool), **Lookup Agent** (`check_appointment_status`), **Response Composer** (merges outputs into a draft).
- Run via `.kickoff()`; demonstrate each tool invoked on different sample queries.
- Implement `MOCK_LLM` by extending `crewai.llms.base_llm.BaseLLM`.

**Known silent pitfalls:**

1. CrewAI's ReAct template contains the literal text `"Observation: the result of the action"`. Do **not** search the whole conversation for `"Observation:"`; parse only the model's own generated text.
2. Do **not** dispatch tool calls by matching tool **name** substrings (a tool named `rag_lookup` would be misclassified). Dispatch off the tool's declared **argument schema**.

### Task 8 — Session memory

- LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` (the `LangChainDeprecationWarning` is expected).
- Transcript A: multi-turn exchange with state carried across turns.
- Transcript B (separate): fresh conversation with state correctly absent.

### Task 9 — Structured output

- Pydantic `BaseModel` for every crew response (`response_format`); validate each response in code.

### Task 10 — Guardrails (each demonstrated firing once)

- **Input:** PII masking on the fixed-format contact number.
- **Input:** prompt-injection detection.
- **Output:** groundedness check that refuses when retrieved context does not support the question.

---

## 7. Part 3 — Evaluation, Observability & Deployment (20 marks)

### Task 11 — FastAPI

- ≥2 HTTP endpoints (e.g., `POST /ask`, `POST /add-document`) with Pydantic request/response models.
- 1 WebSocket endpoint (`@app.websocket`) for multi-turn chat; catch `WebSocketDisconnect` and keep serving other clients.

### Task 12 — Structured logging

- One JSON-Lines entry per request with **trace ID** and **timing**.
- Apply the **same PII masking** to logged text; the raw contact number must never reach disk.

### Task 13 — Evaluation at scale

- LLM-as-judge (under `MOCK_LLM`), **15 queries**: ≥1 per required KB topic (12) + ≥2 out-of-scope/edge cases (+1 more).
- Report **Accuracy, Grounding, Completeness, Safety** per query, plus the **average of each** across all 15.

---

## 8. Part 4 — Resilience & Governance (20 marks)

### Task 14 — Autogen review stage

- 2-agent `RoundRobinGroupChat` (e.g., Policy-Compliance-Reviewer + Final-Editor), bounded with `max_turns=2` (or `MaxMessageTermination(3)`, since it counts the initiating message).
- Final-Editor uses `output_content_type=<VerdictModel>`; the team needs `custom_message_types=[StructuredMessage[VerdictModel]]` or it raises `ValueError: Message type ... is not registered`.
- Verdict model: `approved: bool`, `final_answer: str`, `reason: str`.
- Input: Composer's draft + original retrieved context.
- Demonstrate ≥2 queries: one **approved unchanged**, one **revised** (e.g., a deliberately injected ungrounded claim).

### Task 15 — Four-layer governance

- **Application layer (least autonomy):** only the Lookup Agent may call `check_appointment_status`. Demonstrate that wiring it to another agent is blocked or never wired; include a one-paragraph explanation.
- **Risk classification:** Low / Medium / High with a one-paragraph justification. (Medical data falls under **High**.)
- **Runtime layer:** per-request token/cost budget cap; demonstrate an oversized request being **rejected**, not silently run.

### Task 16 — Response caching

- In-memory cache keyed by **normalized query text** for the grounded-generation step.
- Demonstrate a repeated query producing a hit, with before/after evidence (call counter or timing).

---

## 9. Architecture Overview

```
Client (HTTP / WebSocket)
        │
        ▼
  FastAPI layer ── JSONL logger (trace ID, timing, masked text)
        │
        ▼
  Input guardrails (PII mask, injection detect)
        │
        ▼
  Budget check ── Cache lookup
        │
        ▼
  CrewAI crew (MOCK_LLM via BaseLLM)
   ├─ Retrieval Agent ── RAG tool ── ChromaDB (chosen collection)
   ├─ Lookup Agent ───── check_appointment_status ── dataset.py
   └─ Response Composer ─ draft (Pydantic schema)
        │
        ▼
  Output guardrail (groundedness)
        │
        ▼
  Autogen review (RoundRobinGroupChat, max_turns=2) → Verdict
        │
        ▼
  Final response  (+ session memory per session_id)
```

---

## 10. Suggested Repository Layout

```
.
├── README.md                 # track statement, dataset choices, threshold, env flags
├── ProblemStatement.md
├── requirements.txt
├── dataset.py
├── kb/                       # ≥12 knowledge-base documents
├── rag/
│   ├── chunking.py
│   ├── indexing.py
│   ├── generate.py           # grounded generation + threshold
│   └── evaluate.py           # precision / recall per collection
├── crew/
│   ├── mock_llm.py           # BaseLLM subclass
│   ├── tools.py              # rag tool, check_appointment_status
│   ├── agents.py
│   ├── schemas.py            # Pydantic models
│   ├── guardrails.py
│   └── memory.py
├── review/
│   └── autogen_review.py
├── governance/
│   ├── least_autonomy.py
│   ├── budget.py
│   ├── cache.py
│   └── risk_classification.md
├── api/
│   ├── main.py               # HTTP + WebSocket
│   └── logging_utils.py
├── eval/
│   ├── test_queries.py
│   └── judge.py
└── transcripts/              # one per task (1–16)
```

---

## 11. 14-Day Plan

| Days | Focus | Tasks |
| ------ | ------- | ------- |
| 1 | Setup, env, README skeleton, dataset | 1 |
| 2 | Write knowledge base | 2 |
| 3 | Chunking + embeddings + two ChromaDB collections | 3 |
| 4 | Grounded generation + threshold calibration | 4 |
| 5 | Precision/recall comparison + recommendation | 5 |
| 6 | Escalation-score design + lookup tool | 6 |
| 7–8 | `MOCK_LLM` via `BaseLLM`, 3-agent crew, `.kickoff()` demos | 7 |
| 9 | Session memory, structured output, guardrails | 8, 9, 10 |
| 10 | FastAPI HTTP + WebSocket + JSONL logging | 11, 12 |
| 11 | 15-query LLM-as-judge evaluation | 13 |
| 12 | Autogen review stage | 14 |
| 13 | Governance, budget cap, caching | 15, 16 |
| 14 | Full end-to-end run, transcripts, README polish, submit | all |

---

## 12. Acceptance Criteria Checklist

**Part 1**

- [ ] `dataset.py` meets category counts, status coverage, 10–30% follow-up band
- [ ] README states seed, weights, amount range, fee reasoning
- [ ] ≥12 KB documents cover all required topics
- [ ] Two chunking strategies in two separate ChromaDB collections, sensible retrieval on a sample query
- [ ] ≥5 in-scope queries answered + 1 out-of-scope triggers fallback; measured similarities and threshold in README
- [ ] Per-query precision/recall for both collections + numbers-cited recommendation

**Part 2**

- [ ] `check_appointment_status` returns designed `escalation_score` with stated formula and threshold
- [ ] ≥3-agent crew; both tools invoked on different queries via `.kickoff()`
- [ ] Multi-turn memory transcript + separate fresh-conversation transcript
- [ ] Every response validated against a Pydantic schema
- [ ] PII masking, injection detection, groundedness refusal each demonstrated firing

**Part 3**

- [ ] ≥2 HTTP endpoints + 1 WebSocket surviving client disconnect
- [ ] One JSONL log entry per request with trace ID; no raw PII on disk
- [ ] 15 queries scored on four properties + four averages

**Part 4**

- [ ] Autogen review shows one approved-unchanged and one revised verdict (Pydantic)
- [ ] Least autonomy demonstrated; risk level justified; oversized request rejected by budget cap
- [ ] Cache hit demonstrated with before/after evidence

**Submission**

- [ ] README states **Practo (Healthcare)** track at the top
- [ ] README confirms telemetry disabled and everything runs under `MOCK_LLM` with zero API keys
- [ ] No images, PDFs, slides, video, or audio in the repo
- [ ] One public GitHub link submitted before the deadline

---

## 13. Marks Distribution

| Part | Focus | Marks |
| ------ | ------- | ------- |
| 1 | Dataset Design & RAG Core | 30 |
| 2 | CrewAI Orchestration, Tools, Memory & Guardrails | 30 |
| 3 | Evaluation, Observability & FastAPI Deployment | 20 |
| 4 | Resilience & Governance | 20 |
| | **Total** | **100** |

---

## 14. Risks & Mitigations

| Risk | Mitigation |
| ------ | ------------ |
| Follow-up % falls outside 10–30% | Change seed/weights and regenerate; never hand-edit |
| Untested similarity threshold fails to separate queries | Measure in-scope vs out-of-scope cosine clusters first |
| Mock LLM returns placeholder from ReAct template | Parse only the model's generated text |
| Tool misrouted by name substring | Dispatch on declared argument schema |
| CrewAI telemetry makes a network call | Set `CREWAI_DISABLE_TELEMETRY=true` before kickoff |
| Autogen `ValueError: Message type not registered` | Pass `custom_message_types=[StructuredMessage[Verdict]]` to the team |
| Autogen only one agent speaks | Use `max_turns=2`, or `MaxMessageTermination(3)` |
| Raw phone number reaches logs | Apply the same masker before logging |
| Cache returns stale/incorrect hits | Normalize query text (lowercase, trim, collapse whitespace) consistently |
| Any step silently needs network or API key | Run the full pipeline offline before submitting |
