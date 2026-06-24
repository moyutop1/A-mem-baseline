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

### 1. Temporal Resolver: Event-Matched Candidate Selection

Status: implemented, needs experiment validation

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

Status: implemented, needs experiment validation

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

Status: implemented, needs experiment validation

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

Status: implemented, needs experiment validation

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

Status: implemented, needs experiment validation

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
