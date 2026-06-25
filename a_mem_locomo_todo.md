# A-Mem LoCoMo Iteration Todo List

This file tracks ideas that should not be lost across iterations.

The guiding rule is:

```text
Prefer repairs that strengthen the current main storyline:
evidence-preserving memory -> typed episodic graph -> evidence bundle selection -> answer realization.
```

Do not add scattered category-specific modules unless they are temporary
diagnostic experiments.

---

## Active Next-Step Todos

### Current Active Version: v17_v13_prompt_guard

Status: superseded by v18_rewrite_memory_top10

Reason:

The v16 implementation of profile routing and global short evidence prompts
caused a clear Cat1 regression in the micro-run. The current active code is
therefore restored to the stable v13-style baseline and keeps only conservative
answer-side guards:

```text
stronger Not mentioned instruction
Cat1 rerank fallback when the reranker erases a supported initial answer
```

The five v16 todos below should be treated as implemented once but rolled back
from the active code path until a new diagnostic shows which parts can be
reintroduced without hurting Cat1.

---

### Current Active Version: v18_rewrite_memory_top10

Status: superseded by v19_single_rewrite_index_debug

Reason:

The v17 Cat1/Cat2 results show that answer refusal is no longer the main issue.
The main problem is evidence ranking: Cat1 often hits part of the gold evidence
but fails to rank the full evidence set into the final context.

Implemented repair:

```text
raw dialogue turn
  -> LLM rewrite_content as a self-contained evidence sentence
  -> retrieval, lexical scoring, Cat1 coverage rerank, and graph construction use rewrite_content
  -> final answer prompt still uses original raw memory sentence
  -> final context is capped at top 10 evidence blocks
```

Important constraint:

Do not add new graph edge types for this iteration. The purpose is to test
whether better node text improves ranking without expanding the edge taxonomy.

---

### Current Active Version: v19_single_rewrite_index_debug

Status: active experiment

Reason:

v18 normalized memory content, but the retriever document still repeated
`rewrite_content` three times as an old weighting heuristic. v19 removes that
repetition and adds score diagnostics for debugging low-ranked gold evidence.

Implemented repair:

```text
rewrite_content appears once in _memory_to_index_text
retriever cache version is bumped
candidate_debug stores score_inputs, score_weights, score_contributions
candidate_debug stores combined_rank and source_ranks
candidate_debug output expands to top 100 candidates
```

Important constraint:

This is diagnostic and representation cleanup only. It does not increase the
final evidence context beyond the v18 top-10 setting.

---

### 1. Temporal Resolver: Event-Matched Candidate Selection

Status: rolled back from active v17 code, keep as future candidate

Reason:

The current temporal resolver still tends to use the first temporal candidate
from the ranked context. This can choose the wrong event when multiple retrieved
blocks contain time expressions.

Target repair:

```text
question event phrase
  -> score temporal evidence blocks by event match
  -> choose temporal candidate from the best matched block
```

Expected fixes:

```text
week vs weekend mismatch
wrong pride parade / race / call-out event selected
last year -> anchored year when appropriate
```

Risk:

Medium. It should help Category 2-like temporal questions, but the event matcher
must not become another hand-tuned category rule.

---

### 2. Weak-Inference Answer Realization

Status: partly covered by v17 prompt guard, full structured version rolled back

Reason:

Category 5 showed high evidence coverage but very low judge score, with many
`Not mentioned` predictions. The model is treating weak-inference questions as
strict span extraction.

Target repair:

```text
weak_inference profile
  -> structured support/contradiction evidence list
  -> answer with likely yes/no or likely preference when evidence supports it
```

Expected fixes:

```text
reduce Not mentioned rate
increase LLM judge score for likely / would / might questions
```

Risk:

Medium. Needs careful instruction so the model makes evidence-grounded weak
judgments without hallucinating unsupported preferences.

---

### 3. Fix `infer_answer_type`

Status: rolled back from active v17 code, keep as future candidate

Reason:

The current heuristic overfires on words like `when` even when they appear in a
subordinate clause rather than as the main question intent.

Example:

```text
Who supports Caroline when she has a negative experience?
```

This should be a `person` or support-answer question, not temporal.

Target repair:

```text
Only classify as temporal when the main question asks for time:
- starts with when
- starts with how long
- asks what date / which day
```

Risk:

Low. This is a clear bug fix.

---

### 4. Fix `infer_evidence_need_profile`

Status: rolled back from active v17 code, keep as future candidate

Reason:

The first v15 profile routing was too conservative. Many Category 1-style
multi-evidence questions were routed to `single_span`, reducing context blocks
and lowering `hit_all`.

Examples:

```text
Where has Melanie camped?
What does Melanie do to destress?
What LGBTQ+ events has Caroline participated in?
What desserts has Maria made?
What music events has John attended?
```

Target repair:

```text
route multi-item / attribute bundle questions to multi_item_list or attribute_bundle
keep identity/status/career questions as bundled attribute questions when they
need qualifiers
```

Risk:

Low to medium. The main risk is over-routing true single-span questions into a
larger evidence bundle, increasing noise.

---

### 5. Short Evidence List Before LLM Answering

Status: rolled back from active v17 code, keep as future candidate

Reason:

Prompt length is still high, but v15 showed that simply reducing the number of
context blocks hurts recall. The better direction is to keep enough evidence
blocks while shortening each block.

Target repair:

Convert raw retrieved context into structured, compact evidence records:

```text
[Evidence 1]
dia_id:
session_date:
speaker:
fact:
image_caption:
relation:
why_selected:
```

Expected benefit:

```text
keep 12-16 evidence candidates
reduce prompt tokens
make answer extraction more stable
preserve dia_id-level diagnostics
```

Risk:

Medium. Compression must preserve answer-bearing details such as dates, names,
lists, image query facts, and qualifiers.

---

## Deferred Research Todo

### Summary/Event Nodes for Graph Support

Status: deferred

Reason:

The current system stores many raw dialogue fragments. A compact event-level
representation may help graph construction and routing, but it must not replace
raw evidence.

Important constraint:

Do not add many new persistent edge types. Earlier experiments showed that
adding more edge types can hurt retrieval by increasing noisy graph expansion.

Preferred design:

```text
raw dialogue turn nodes remain the only final evidence source
event summary nodes are auxiliary routing / graph support nodes
event summary nodes point back to raw evidence ids
final answer prompt expands raw evidence, not summaries only
```

Possible implementation:

```text
adjacent raw turns
  -> model creates one event summary node
  -> summary stores evidence_ids = [D5:1, D5:2, ...]
  -> existing relation types connect summaries or guide raw-node linking
  -> retrieval uses summary only to find which raw evidence bundle to expose
```

Allowed relation policy:

Use existing relation semantics where possible:

```text
same_storyline
similar_event
image_text_pair
same_character
```

Avoid adding a large new edge taxonomy unless a diagnostic experiment proves it
improves `hit_all` and judge score.

Risk:

High enough to defer. It may reduce prompt length and improve graph stability,
but it may also hide details that LoCoMo answers require.

---

## Diagnostic Metrics To Keep Reporting

```text
hit_any
hit_all
hit_all_but_wrong
miss_all_but_wrong
high_f1_judge_wrong
not_mentioned_rate
empty_answer_rate
answer_type_distribution
evidence_profile_distribution
mean prompt chars
mean raw context chars
mean context block count
```
