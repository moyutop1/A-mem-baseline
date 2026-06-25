# A-Mem LoCoMo Modification Plan: Innovation Points and System Flow

## 0. Goal

This document summarizes the planned A-Mem modification for LoCoMo-style precise episodic QA.

The current priority is to improve evidence recall and reduce noisy memory interference. The system should preserve raw episodic evidence, retrieve a small and accurate memory bundle, and use bundle-level reflection to update retrieval behavior.

We focus on five innovation points:

1. Evidence-preserving Episodic Memory Writer
2. Episodic Memory Graph Retrieval
3. Soft Domain Rerank + Hybrid Evidence Retrieval
4. Memory-bundle-level Reflection
5. Memory Life-cycle Utility Management

Temporal normalization is not an independent innovation point. It is treated as a supporting component inside graph retrieval and reranking.

---

## 1. Problem Diagnosis

Early LoCoMo experiments show that the main failure is evidence retrieval, not only answer generation.

Example:

```text
Q: What did Caroline research?
Predicted: Caroline did not research anything.
Gold: Adoption agencies
Gold evidence: Researching adoption agencies...
```

This suggests that the gold evidence did not enter the final context, or it was overwhelmed by irrelevant retrieved memories.

Another failure type is temporal QA:

```text
Q: When did Caroline go to the LGBTQ support group?
Predicted: last Tues
Gold: 7 May 2023
Evidence: In an 8 May 2023 session, the dialogue says "yesterday".
```

This requires explicit temporal normalization:

```text
8 May 2023 + yesterday = 7 May 2023
```

Top-k retrieval also introduces noise. Increasing k may retrieve semantically similar but incorrect memories, such as book/education-related memories that distract from the correct answer "psychology, counseling certification".

---

## 2. Innovation A: Evidence-preserving Episodic Memory Writer

### Core Idea

For LoCoMo, each raw dialogue turn should be stored as an immutable instance-level evidence memory.

Do not aggressively rewrite, merge, or summarize raw episodic turns.

### Required Memory Fields

Each LoCoMo memory node should include:

```yaml
memory_id: locomo_D2_8
conversation_id: conv_001
dia_id: D2:8
session_id: D2
turn_id: 8
session_date: 2023-05-25
speaker: Caroline
content: "Researching adoption agencies..."
raw_text: "Caroline: Researching adoption agencies..."
memory_level: instance
rewrite_allowed: false
status: active_evidence
entities:
  - Caroline
  - adoption agencies
temporal_expressions: []
```

For temporal memories:

```yaml
memory_id: locomo_D3_12
dia_id: D3:12
session_date: 2023-05-08
speaker: Caroline
content: "I went to the LGBTQ support group yesterday."
raw_text: "Caroline: I went to the LGBTQ support group yesterday."
memory_level: instance
rewrite_allowed: false
status: active_evidence
entities:
  - Caroline
  - LGBTQ support group
temporal_expressions:
  - text: yesterday
    anchor: session_date
    normalized_date: 2023-05-07
```

### Rule

```python
if dataset == "LoCoMo" and memory_level == "instance":
    create_separate_memory()
    rewrite_allowed = False
    allow_link = True
```

Raw LoCoMo instance memories should not be rewritten. They can only be linked, retrieved, grounded, or used as evidence.

---

## 3. Innovation B: Episodic Memory Graph Retrieval

### Core Idea

Do not retrieve memories as isolated top-k text chunks. Instead, treat raw dialogue turns as evidence nodes in an episodic memory graph.

The graph should help retrieve local context, same-event evidence, speaker-aligned memories, entity-aligned memories, and temporal anchors.

### Disallowed Edge Types

The following edge types should not exist as persistent graph edges:

```text
updates
replaces
conflicts_with
```

Updates and conflicts should be resolved through memory writing or reflection procedures, not graph traversal.

### Allowed Edge Types

#### 1. `adjacent_turn`

Connect nearby turns in the same session.

```text
D2:7 <-> D2:8
D2:8 <-> D2:9
```

Purpose: local context expansion.

#### 2. `same_session`

Memories from the same session.

Purpose: weak bonus or candidate pool. Do not expand the entire session into the final context.

#### 3. `same_speaker`

Memories from the same speaker.

Purpose: query-time speaker bonus. Do not globally expand all same-speaker memories.

#### 4. `mentions_same_entity`

Connect memories that mention the same entity, object, activity, or noun phrase.

Examples:

```text
adoption agencies
LGBTQ support group
psychology
counseling certification
```

#### 5. `same_event`

Connect nearby memories that likely describe the same event.

A simple first-version rule:

```python
if same_session(i, j) and abs(turn_i - turn_j) <= 3:
    if shared_entity(i, j) or high_lexical_overlap(i, j):
        add_edge(i, j, "same_event", weight=0.8)
```

#### 6. `temporal_anchor`

A memory contains a relative temporal expression linked to the session date.

This can be stored as metadata rather than a persistent edge:

```yaml
temporal_expressions:
  - text: yesterday
    anchor: 2023-05-08
    normalized_date: 2023-05-07
```

#### 7. `evidence_supports_summary`

If a derived or summary memory exists, it must be grounded in raw evidence memories.

```text
raw_instance_memory -> summary_memory
```

### Edge Storage

Use an independent edge table or graph store.

```sql
CREATE TABLE memory_edges (
    edge_id TEXT PRIMARY KEY,
    src TEXT,
    dst TEXT,
    edge_type TEXT,
    weight REAL,
    metadata_json TEXT
);
```

Example:

```json
{
  "edge_id": "edge_0001",
  "src": "locomo_D2_8",
  "dst": "locomo_D2_9",
  "edge_type": "adjacent_turn",
  "weight": 1.0,
  "metadata": {
    "distance": 1,
    "direction": "next"
  }
}
```

### Graph Retrieval Flow

```text
query
  ↓
question type classification
  ↓
hybrid seed retrieval
  ↓
question-type-specific edge expansion
  ↓
evidence-focused reranking
  ↓
small role-labeled memory bundle
```

Question types may include:

```text
factual_action
temporal
preference_or_likely
relationship_identity
summary
```

Example edge policy:

```python
edge_policy = {
    "factual_action": ["adjacent_turn", "mentions_same_entity"],
    "temporal": ["same_event", "adjacent_turn", "mentions_same_entity"],
    "preference_or_likely": ["mentions_same_entity", "same_speaker", "same_event"],
    "relationship_identity": ["mentions_same_entity", "adjacent_turn", "same_session"],
}
```

Limit graph expansion. Retrieve many candidates, but select only a small final bundle.

Suggested first-version limits:

```yaml
seed_top_k: 20
graph_expand_per_seed: 2
final_context_budget: 4-6
max_adjacent_turns: 2
```

---

## 4. Innovation C: Soft Domain Rerank + Hybrid Evidence Retrieval

### Core Idea

Domain routing should not be a hard filter.

Wrong domain routing can block gold evidence. Therefore, domain information should be used only as a soft reranking bonus.

### Scoring

A simple first-version score:

```text
score(m) =
  embedding_score(q, m)
+ BM25_score(q, m)
+ speaker_entity_score(q, m)
+ edge_score(q, m)
+ temporal_score(q, m)
+ domain_bonus(q, m)
- noise_penalty(q, m)
```

Domain bonus should only increase score. It should not remove non-matching memories.

A lightweight version:

```text
score(m) = s_embed(q, m) + 0.15 * I(domain_match) - 0.20 * lifecycle_penalty(m)
```

### Hybrid Candidate Retrieval

Use multiple candidate sources:

```python
candidates = []
candidates += embedding_retrieve(query, all_memories, top_k=20)
candidates += bm25_retrieve(query, all_memories, top_k=20)
candidates += speaker_entity_retrieve(query, all_memories, top_k=20)

if is_temporal_question(query):
    candidates += temporal_retrieve(query, all_memories, top_k=10)

if domain_router_confident:
    candidates += domain_biased_retrieve(query, top_domains, top_k=10)

candidates = deduplicate(candidates)
ranked = evidence_rerank(candidates)
final_bundle = select_top_evidence_bundle(ranked, budget=4_to_6)
```

### Principle

```text
retrieve many candidates, select few evidence memories
```

Do not simply increase final top-k. Larger top-k can increase noise and reduce answer accuracy.

---

## 5. Innovation D: Memory-bundle-level Reflection

### Core Idea

Reflection should not only ask whether each single memory was useful. It should evaluate whether the retrieved memory bundle as a whole helped reasoning.

The system should diagnose whether the bundle contained:

```text
missing evidence
noisy memories
redundant memories
stale or outdated memories
conflicting memories
ungrounded summaries
inconsistency between derived memories and raw instance evidence
memories repeatedly retrieved but rarely helpful
```

### Reflection Flow

```text
query
  ↓
retrieve memory bundle
  ↓
agent uses memory bundle
  ↓
execution / response feedback
  ↓
bundle-level reflection
  ↓
update memory states, retrieval weights, graph edge weights, and grounding links
```

### Edit Actions

The reflection module should output structured edit actions.

#### ADD

Necessary evidence is missing.

Actions:

```text
trigger supplementary retrieval
log missed evidence
create new memory only if evidence is absent from the memory store
```

#### PRUNE

A memory was retrieved but did not help or introduced noise.

Actions:

```text
lower retrieval weight for this memory under the current question type / query pattern
do not delete raw evidence memory
```

#### SUBSTITUTE

A broad or noisy memory should be replaced by a more precise evidence memory.

Example:

```text
replace summary memory with raw instance evidence
```

#### MERGE

Multiple related instance memories can support a derived memory.

Important constraint:

```text
do not physically merge or delete raw instance memories
create a derived memory and preserve evidence links
```

#### SPLIT

A mixed or overly broad derived memory should be split into cleaner memory notes.

Mostly applies to:

```text
summary memory
generated memory
badly constructed memory note
```

Not usually raw LoCoMo turns.

#### CONTRADICT

The bundle contains inconsistency.

Important constraint:

```text
CONTRADICT is a reflection action, not a persistent graph edge.
```

For raw instance memories, do not overwrite or delete them. For derived memories, trigger resolution, rewriting, or grounding correction.

#### ABSTRACT

Create a generalized memory from multiple raw evidence memories.

Constraint:

```text
ABSTRACT must preserve supporting evidence links.
```

#### GROUND

A summary or generalized memory must be linked to supporting raw evidence.

If no supporting evidence exists:

```text
status = ungrounded_summary
retrieval_weight = low
```

### Reflection Output Format

Example:

```json
{
  "bundle_helpfulness": "insufficient",
  "diagnosis": [
    {
      "issue": "missing_evidence",
      "description": "The bundle did not include the memory where Caroline said she was researching adoption agencies."
    },
    {
      "issue": "retrieval_noise",
      "description": "Retrieved memories about school/books are semantically related but do not answer the research question."
    }
  ],
  "edit_actions": [
    {
      "action": "ADD",
      "target": "locomo_D2_8",
      "reason": "Gold evidence should be retrieved for factual_action questions involving Caroline and research."
    },
    {
      "action": "PRUNE",
      "target": "locomo_D4_2",
      "scope": {
        "question_type": "factual_action",
        "query_terms": ["research"]
      },
      "reason": "This memory was retrieved but did not contribute to the answer."
    }
  ],
  "retrieval_policy_update": {
    "increase": ["BM25", "speaker_match", "lexical_trigger"],
    "decrease": ["broad_semantic_related", "same_session"]
  }
}
```

---

## 6. Innovation E: Memory Life-cycle Utility Management

### Core Idea

Memory retrieval should not treat every stored memory as equally useful forever.

Each raw evidence memory should maintain life-cycle utility statistics from retrieval and answer feedback. Memories that are repeatedly retrieved and repeatedly help produce correct answers should receive a positive rerank bonus. Memories that are often retrieved but rarely help, introduce noise, or lead to wrong answers should receive a penalty under similar query types.

This innovation is not a memory rewriting mechanism. It is a retrieval policy adaptation mechanism.

The key idea is broader than "frequently retrieved memories become more important".
Retrieval frequency alone is not a reliable utility signal. A memory should become more important only when it is repeatedly useful under a specific query condition, and it should become less important when it is repeatedly retrieved as noise.

Therefore, life-cycle management should model memory utility as a multi-dimensional retrieval history:

```text
memory utility
  = usefulness for this question type
  + usefulness for this domain / entity / storyline
  + grounding reliability
  + recency / freshness when relevant
  - noise history
  - redundancy with stronger evidence
  - stale or ungrounded summary penalty
```

This module is intended to be a later-stage innovation after evidence recall becomes acceptable. If early retrieval is unstable, life-cycle feedback can reinforce retrieval mistakes.

### Motivation

In LoCoMo-style long conversations, many memories are semantically related to a question but are not useful evidence for answering it.

Example:

```text
Question: What did Caroline research?
Useful evidence: adoption agencies
Noisy evidence: books, school, general identity discussion
```

If a noisy memory is repeatedly retrieved for research-related questions and repeatedly fails to support the correct answer, the system should learn to reduce its retrieval priority for that query pattern.

Similarly, if a memory such as `D2:8` repeatedly supports correct answers about Caroline researching adoption agencies, it should become easier to retrieve for future similar questions.

### Required Life-cycle Fields

Each memory should maintain lightweight feedback statistics:

```yaml
retrieval_count: 0
citation_count: 0
successful_citation_count: 0
failed_citation_count: 0
noise_count: 0
missed_when_gold_count: 0
last_retrieved_at: null
last_successful_at: null
life_cycle_status: active
utility_by_question_type:
  factual_action:
    successful: 0
    failed: 0
  temporal:
    successful: 0
    failed: 0
  multi_evidence_list:
    successful: 0
    failed: 0
utility_by_domain:
  "Personal Life / Identity":
    successful: 0
    failed: 0
utility_by_entity:
  Caroline:
    successful: 0
    failed: 0
utility_by_storyline:
  "Caroline LGBTQ participation":
    successful: 0
    failed: 0
redundancy_count: 0
stale_count: 0
grounding_success_count: 0
grounding_failure_count: 0
exploration_count: 0
```

Raw LoCoMo memories remain immutable evidence nodes. Life-cycle management updates metadata and retrieval weights only.

### Confirmed First-version Design

The first implementation should use the following simplified design.

```text
1. Use a global utility score for each memory.
2. Share this utility across the current memory tree / domain tree.
3. Update life-cycle statistics after every answered question.
4. Use an LLM judge to classify each retrieved memory as useful, partially useful, redundant, noisy, irrelevant, or misleading.
5. If the answer is wrong, penalize only memories judged noisy, irrelevant, or misleading.
6. Do not penalize redundant memories; record redundancy_count only.
7. Use gold evidence dia_ids only for research diagnostics and evaluation analysis, not as a required signal in the general system.
```

This keeps the first version simple and avoids prematurely adding question-type-specific utility before retrieval behavior is stable.

### Multi-dimensional Utility Score

A first-version Bayesian utility estimate:

```text
utility(m) =
  (successful_citation_count + alpha)
  /
  (successful_citation_count + failed_citation_count + alpha + beta)
```

Suggested default:

```text
alpha = 1
beta = 1
```

This avoids over-trusting memories after only one successful retrieval.

The first-version rerank score can include:

```text
score(m, q) =
  evidence_relevance(m, q)
+ domain_bonus(m, q)
+ graph_bonus(m, q)
+ lambda_utility * global_utility(m)
- lambda_noise * global_noise_penalty(m)
- lifecycle_penalty(m)
```

For a stronger later version, utility can become scoped rather than global:

```text
utility(m, q) =
  w_global * global_utility(m)
+ w_type * utility_by_question_type(m, type(q))
+ w_domain * utility_by_domain(m, routed_or_predicted_domain(q))
+ w_entity * utility_by_entity(m, entities(q))
+ w_story * utility_by_storyline(m, storyline(q))
+ w_ground * grounding_reliability(m)
- w_noise * noise_rate(m, type(q))
- w_redundant * redundancy_penalty(m, current_bundle)
```

Scoped utility can prevent a memory from becoming globally over-promoted just because it helped one narrow question type. This is a later extension, not the first implementation.

### Additional Life-cycle Signals

The following signals can make this module more substantial than a single success-count feature.

#### 1. Query-type-scoped utility

The same memory may be useful for temporal questions but noisy for list-style questions.

Example:

```text
D5:1 pride parade
Useful for: LGBTQ event questions
Possibly noisy for: generic identity questions
```

First-version decision:

```text
Do not implement this yet. Start with global utility.
```

#### 2. Domain/entity/storyline utility

Track whether a memory is useful for a specific domain, entity, or storyline.

Example:

```text
Storyline: Caroline LGBTQ participation
Useful memories: support group, school speech, pride parade, mentorship, art show
```

This can later interact with graph expansion. High-utility memories can become stronger seeds for the corresponding storyline.

First-version decision:

```text
Do not implement scoped domain/entity/storyline utility yet. Keep these fields as future extensions.
```

#### 3. Grounding reliability

Derived or summary memories should receive utility only if they are consistently grounded in raw evidence.

Raw LoCoMo turn memories start with high grounding reliability because they are direct evidence. Generated summaries must earn grounding reliability through supporting evidence links.

#### 4. Noise and redundancy tracking

A memory should be penalized if it is often retrieved but does not contribute to the final answer.

Redundancy should be treated separately from noise:

```text
noise: irrelevant or misleading evidence
redundancy: relevant but adds no new answer slot beyond stronger evidence
```

This matters for bundle selection. A relevant but redundant memory should not be deleted or globally penalized; it should simply lose priority when the current bundle already contains equivalent evidence.

First-version decision:

```text
Noise is judged by an LLM after each answer.
Penalize only memories judged noisy, irrelevant, or misleading.
Redundant memories are not penalized globally; only record redundancy_count.
```

#### 5. Freshness and staleness

Some memories become stale when newer evidence changes the user's state, plan, or preference.

For LoCoMo raw evidence, the old memory should remain preserved, but retrieval should prefer the newer memory when the question asks about current status.

Example:

```text
Older memory: considering a plan
Newer memory: decided on a plan
Current-status question: prefer newer evidence
Historical question: keep older evidence retrievable
```

#### 6. Exploration vs exploitation

If the system always boosts previously successful memories, it may overfit early feedback and repeatedly retrieve the same popular evidence.

Use a small exploration mechanism:

```text
mostly retrieve high-utility memories
occasionally allow uncertain but relevant memories into the candidate pool
```

This is especially important before the system has enough feedback.

### Positive Feedback

Increase utility when:

```text
1. The memory appears in the final evidence bundle.
2. The LLM judge marks the memory as useful or directly supporting.
3. The final answer is correct or has high F1.
4. During research evaluation, the memory's dia_id overlaps with gold evidence.
```

Gold evidence dia_id overlap is an evaluation shortcut, not a required signal for real deployment.

### Negative Feedback

Decrease utility when:

```text
1. The final answer is wrong or low-quality.
2. The LLM judge marks the retrieved memory as noisy, irrelevant, or misleading.
3. The memory repeatedly appears in wrong-answer bundles as judged noise.
4. The memory repeatedly displaces stronger evidence under similar query patterns.
```

Do not penalize every memory in a wrong-answer bundle. Some retrieved memories may be correct gold evidence, and the failure may come from answer extraction rather than retrieval.

### Life-cycle Status

Possible statuses:

```text
active
high_utility
low_utility
noisy_for_query_type
stale
needs_grounding
```

Important constraint:

```text
status affects retrieval priority only.
It must not delete or rewrite raw LoCoMo evidence memories.
```

### Relationship to Bundle Reflection

Bundle-level reflection produces the feedback signal. Life-cycle management stores and applies the signal.

```text
retrieved bundle
  -> answer
  -> evaluation / reflection
  -> identify useful, missing, noisy, redundant memories
  -> update memory utility statistics
  -> future rerank uses learned utility
```

Life-cycle updates should be applied after bundle-level diagnosis, not directly from final answer correctness alone.

First-version update timing:

```text
After every answered question:
  1. evaluate answer quality
  2. ask LLM to judge each retrieved memory's role
  3. update global utility / noise / redundancy statistics
  4. immediately use updated utility for the next question
```

Example:

```text
Wrong answer + retrieved memory is actually gold evidence:
  do not penalize the memory
  diagnose answer extraction failure

Wrong answer + retrieved memory is unrelated:
  penalize the memory for the current question type / query pattern

Correct answer + memory overlaps gold evidence:
  increase global utility

Correct answer + memory was retrieved but unused:
  do not increase utility
  possibly mark as redundant
```

### LLM Judge Output for Life-cycle Updates

The LLM judge should classify each retrieved memory after an answer.

Example output:

```json
{
  "answer_quality": "incorrect",
  "memory_judgments": [
    {
      "dia_id": "D2:8",
      "role": "useful",
      "reason": "This memory directly states that Caroline researched adoption agencies.",
      "utility_update": "increase"
    },
    {
      "dia_id": "D8:37",
      "role": "noisy",
      "reason": "This memory discusses a book and distracts from the adoption-agency answer.",
      "utility_update": "decrease"
    },
    {
      "dia_id": "D8:38",
      "role": "redundant",
      "reason": "This memory is related to the same broad topic but adds no new answer evidence.",
      "utility_update": "none"
    }
  ]
}
```

Allowed memory roles:

```text
useful
partially_useful
redundant
noisy
irrelevant
misleading
```

Update policy:

```text
useful / partially_useful + correct answer:
  increase successful_citation_count

noisy / irrelevant / misleading + wrong answer:
  increase failed_citation_count and noise_count

redundant:
  increase redundancy_count only
```

### Evaluation for Life-cycle Management

This innovation should be evaluated beyond final QA F1.

Suggested metrics:

```text
Utility-weighted Evidence Recall
Noise Memory Retrieval Rate
Repeated-noise Suppression Rate
Gold Evidence Promotion Rate
High-utility Memory Precision
Low-utility Memory Suppression Accuracy
Answerable Context Rate before/after utility updates
Performance over repeated evaluation rounds
```

Suggested ablation:

```text
without life-cycle utility
global utility only
question-type-scoped utility
domain/entity/storyline-scoped utility
full utility + redundancy/noise penalty
```

First-version ablation should prioritize:

```text
without life-cycle utility
global utility only
global utility + LLM noise penalty
```

The expected claim is not only "higher F1", but:

```text
The memory system learns which memories are useful evidence under specific query conditions,
improves repeated retrieval quality, and suppresses recurring noisy memories without rewriting
or deleting raw episodic evidence.
```

### Deferred Questions Before Advanced Implementation

The following details are deferred to later versions:

```text
1. Whether utility should later become scoped by question type / domain / query pattern.
2. How to replace gold evidence dia_id diagnostics with pure reflection signals in non-labeled settings.
3. How strongly utility should affect reranking compared with BM25, dense similarity, domain bonus, and graph score.
4. How to prevent early online feedback noise from permanently suppressing useful memories.
5. How to measure redundancy separately from irrelevance at scale.
6. Whether to add exploration bonuses for uncertain but relevant memories.
```

This module should be implemented only after evidence retrieval and bundle selection reach a reasonable level, so that life-cycle updates are based on meaningful feedback rather than unstable early retrieval behavior.

---

## 7. Temporal Normalization as Supporting Component

Temporal normalization is not a standalone innovation point. It supports Innovations B and C.

### Required Functionality

Detect relative temporal expressions:

```text
yesterday
last Saturday
last week
next month
last year
```

Normalize using `session_date`.

Examples:

```text
8 May 2023 + yesterday = 7 May 2023
25 May 2023 + last Saturday = Saturday before 25 May 2023
9 June 2023 + last week = the week before 9 June 2023
```

### Prompt Formatting

For temporal QA, include normalized time in the evidence bundle.

```text
[Evidence]
Source: D3:12
Session Date: 8 May 2023
Speaker: Caroline
Text: "I went yesterday."
Temporal Normalization:
- "yesterday" relative to 8 May 2023 = 7 May 2023
```

---

## 7. Full System Flow

### 7.1 Memory Writing

```text
LoCoMo dialogue turn
  ↓
extract dia_id / session_id / turn_id / session_date / speaker / raw_text
  ↓
create immutable instance memory
  ↓
extract entities / noun phrases
  ↓
extract temporal expressions
  ↓
normalize temporal expressions when possible
  ↓
build stable episodic graph edges
```

### 7.2 Retrieval and Answering

```text
query
  ↓
question type classification
  ↓
hybrid seed retrieval
  - embedding
  - BM25
  - speaker match
  - entity match
  - temporal cue
  - soft domain bonus
  ↓
episodic graph expansion
  ↓
evidence-focused reranking
  ↓
select small role-labeled memory bundle
  ↓
format prompt with source ids, speakers, dates, relation reasons, and temporal normalization
  ↓
LLM answer
```

### 7.3 Post-answer Reflection

```text
query + retrieved bundle + raw_context + prediction + feedback
  ↓
bundle-level reflection
  ↓
diagnose bundle issues
  ↓
generate edit actions
  ↓
update retrieval weights, graph weights, memory states, grounding links, and diagnostic logs
```

---

## 8. Required Logging for Debugging and Evaluation

Each individual result should save:

```json
{
  "query": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "raw_context": "...",
  "user_prompt": "...",
  "retrieved_memory_ids": ["..."],
  "retrieved_dia_ids": ["..."],
  "used_edges": ["..."],
  "question_type": "...",
  "temporal_normalizations": ["..."],
  "reflection_output": {}
}
```

This is required to diagnose:

```text
whether gold evidence entered raw_context
whether the context was answerable
whether noise overwhelmed the evidence
whether temporal normalization was correct
```

---

## 9. Suggested Ablation Plan

Run incremental ablations:

```text
B0: Original A-Mem
B1: + raw_context / user_prompt / retrieved IDs logging
B2: + evidence-preserving LoCoMo memory writer
B3: + hybrid retrieval + soft domain rerank
B4: + episodic graph retrieval
B5: + temporal normalization support
B6: + memory-bundle-level reflection
```

Important comparisons:

```text
hard domain filter vs soft domain bonus
flat top-k vs retrieve-many-select-few
embedding-only vs hybrid retrieval
with vs without graph expansion
with vs without temporal normalization
with vs without bundle-level reflection
rewrite instance memory vs no rewrite
```

---

## 10. Evaluation Metrics

Final QA metrics:

```text
Exact Match
F1
ROUGE-1 F
METEOR
SBERT similarity
BERTScore if available locally
```

Retrieval diagnostics:

```text
Evidence Recall@K
Answerable Context Rate
Gold Evidence Rank
Noise Ratio
Temporal Resolution Accuracy
Raw Instance Preservation Rate
Rewrite Harm Rate
Bundle Helpfulness Rate
```

These diagnostics are important because final QA metrics alone cannot explain whether failures come from retrieval, temporal reasoning, or answer generation.

---

## 11. Implementation Priorities

### Phase 1: Must-have

```text
save raw_context and user_prompt
save retrieved_memory_ids and dia_ids
write LoCoMo turns with dia_id / session_date / speaker / raw_text
disable rewrite for instance memories
```

### Phase 2: Retrieval improvement

```text
add BM25 retrieval
add speaker/entity retrieval
replace hard domain filtering with soft domain bonus
select final context budget 4-6
```

### Phase 3: Graph retrieval

```text
build adjacent_turn edges
build mentions_same_entity edges
build same_event edges
use question-type-specific edge expansion
```

### Phase 4: Temporal support

```text
detect relative temporal expressions
normalize with session_date
include normalized dates in evidence bundle prompt
```

### Phase 5: Bundle reflection

```text
log bundle-level feedback
implement ADD / PRUNE / SUBSTITUTE / GROUND first
later add MERGE / SPLIT / CONTRADICT / ABSTRACT
```

---

## 12. Constraints

The following constraints must be preserved:

```text
1. Raw LoCoMo instance memories must not be rewritten.
2. Updates/replaces/conflicts_with are not persistent graph edges.
3. Domain routing must not hard-filter the memory pool.
4. Final context should be a small evidence bundle, not a large flat top-k list.
5. Summary or derived memories must be grounded in raw evidence.
6. CONTRADICT is a reflection action, not a graph edge.
7. MERGE and ABSTRACT may create derived memories, but must not delete raw evidence.
```

---

## 13. One-sentence Summary

We modify A-Mem into an evidence-first episodic memory system for LoCoMo: raw dialogue turns are preserved as immutable evidence nodes; retrieval combines hybrid seed search, soft domain reranking, and episodic graph expansion; the final prompt receives a small role-labeled evidence bundle with source ids and temporal normalization; after answering, bundle-level reflection diagnoses missing evidence, noise, redundancy, ungrounded summaries, and inconsistent derived memories, then updates retrieval weights, graph weights, memory states, and grounding links without rewriting raw instance memories.

---

## 14. v12 Typed Graph Simplification

### Motivation

The v9 typed graph experiments showed that most graph contribution came from `graph_similar_event`, while newly introduced typed edges rarely became dominant retrieval evidence. The likely reason is not only the edge idea itself, but the implementation:

```text
- Too many public relation types made the graph difficult to interpret and ablate.
- Some relations encoded broad answer slots rather than concrete evidence chains.
- Newly introduced typed relations were mostly used as weak candidate scores, not explicitly shown in the final answer context.
- Storyline tags such as books_music, art, and pets_items were too coarse for LoCoMo evidence matching.
```

### Retained Public Edge Types

v12 uses a smaller public edge vocabulary:

```text
local_context
similar_event
same_character
same_storyline
image_text_pair
```

The following v9 edge types are no longer generated as public offline graph relations:

```text
same_answer_slot
shared_activity
shared_artifact
temporal_followup
before_after
local_evidence_pair
supports
derived_from
clarifies_answer
```

Their useful information is folded into `same_storyline` as concrete cue evidence and edge reason text.

### Deterministic Storyline Cue Extraction

v12 does not use an LLM to extract storyline cues in the main graph construction path. This keeps the graph deterministic, cheaper to rebuild, and easier to ablate.

Each memory now gets a concrete cue profile:

```python
{
    "speaker": speaker,
    "storyline_cues": set(...),
    "strong_cues": set(...),
    "visual_cues": set(...),
    "cue_types": {
        "person": set(...),
        "visual": set(...),
        "object": set(...),
        "activity": set(...),
    },
}
```

Cue sources:

```text
speaker
content
image_caption
image_query
memory context
keywords
tags
domain paths
```

Cue extraction rules:

```text
- quoted phrases
- capitalized spans
- keyword/tag/domain short phrases
- ordered bigrams/trigrams from content tokens
- protected high-value phrases such as mental health, adoption agency, single parent, LGBTQ support group
- visual cues from image caption/query when present
```

The extractor filters structural and conversational noise:

```text
dia, speaker, session, content, conversation, memory, image, photo,
you, what, how, good, thanks, cool, wow, support, story, etc.
```

### New `same_storyline` Rule

`same_storyline` is created from concrete shared cues, not broad tags.

```text
same speaker + shared strong cue >= 1 -> same_storyline
same speaker + shared non-person cue >= 2 -> same_storyline
different speaker + shared strong cue >= 2 -> same_storyline
chronological cross-session pair + shared strong cue -> same_storyline
```

Edge reason explicitly records the cue evidence:

```text
Shared concrete storyline cues: adoption, adoption agency, agency
Shared concrete storyline cues: mental health; chronological follow-up across sessions
```

### New `image_text_pair` Rule

`image_text_pair` is only created when an image-derived cue overlaps another memory's storyline cue within nearby same-session turns:

```text
same session nearby
visual_cues(left) overlap storyline_cues(right)
or visual_cues(right) overlap storyline_cues(left)
```

This replaces the earlier broad rule that linked nearby turns whenever an artifact tag was present.

### Final Context Exposure

v12 exposes typed graph structure to the answer LLM instead of hiding it inside retrieval scores.

Final context can now include:

```text
relation: same_storyline
edge reason: Shared concrete storyline cues: adoption, adoption agency, agency
talk start time: ...
memory content: ...
```

This is intended to make graph expansion interpretable and measurable in result JSON files.

### Expected Diagnostics

The v12 run should be judged by both retrieval and answer metrics:

```text
relation_counts.same_storyline in raw_context
graph_same_storyline in candidate_debug.source_tags
gold evidence rescued by same_storyline expansion
evidence_hit_any / evidence_hit_all
F1 / EM
Not-mentioned rate by question type
```

If `same_storyline` improves evidence_hit_all but F1 does not improve, the remaining bottleneck is answer generation rather than graph retrieval.

---

## 14. v14 Diagnostic Patch: Category 2 Temporal Resolver

### Motivation

The v13 evidence-first run shows that Category 2 retrieval is already strong:

```text
category 2 evidence_hit_any: 77 / 90
category 2 evidence_hit_all: 75 / 90
category 2 LLM judge score: 0.600
```

This means many remaining Category 2 failures are not caused by missing evidence.
They are caused by temporal answer realization:

```text
evidence is retrieved
relative temporal expression is present
the answer model outputs an empty answer, "Not mentioned", or the wrong anchored date
```

Therefore v14 adds a small deterministic temporal resolver as a diagnostic
upper-bound experiment for Category 2 only.

### Important Scope Note

This patch intentionally uses `category == 2` as a temporary diagnostic gate.
It should not be presented as the final inference-time design, because LoCoMo
category labels are benchmark metadata.

The intended final design is:

```text
question text
  -> infer evidence requirement / question type
  -> if temporal reasoning is required, use temporal resolver
```

The current Category 2 gate is used only to test whether temporal resolving is
worth keeping before converting it into question-type-aware routing.

### Superseded by v15 Cleanup

The temporary `category == 2` gate has now been removed from the code. The
temporal resolver is kept only as a question-type component:

```text
answer_type == temporal
  -> resolve_temporal_answer(...)
```

This preserves the useful v14 diagnostic gain while avoiding benchmark-label
routing in the final method.

### Implementation Location

Implemented in:

```text
test_advanced_robust.py
```

Main functions:

```text
_temporal_candidates_from_block
resolve_temporal_answer
```

Activation path:

```python
if answer_type == "temporal":
    response = self.resolve_temporal_answer(response, raw_context, question)
```

This keeps Category 1, Category 4, and Category 5 behavior unchanged.

### Candidate Generation

For each high-ranked temporal evidence block, the resolver creates structured
temporal candidates:

```json
{
  "raw_expression": "last Saturday",
  "selected_answer": "The Saturday before 25 May 2023",
  "answer_style": "anchored_relative",
  "absolute_date": "",
  "session_date": "25 May 2023",
  "dia_id": "D2:1",
  "raw_fact": "..."
}
```

Supported first-version patterns:

```text
yesterday / today / tomorrow
last week / the week before
last weekend / previous weekend / the weekend before
last Monday ... last Sunday
last month / this month / next month
last year
since YYYY
for N years
explicit date, when the question asks for a date
bare year expressions such as "in 2022"
```

### Relative vs Absolute Date Policy

The resolver stores both absolute and LoCoMo-style verbalizations when useful,
but it usually returns benchmark-style anchored relative answers for weekday,
week, and weekend expressions:

```text
last Saturday + session date 25 May 2023
-> The Saturday before 25 May 2023
```

For expressions like `yesterday`, it returns the resolved absolute date:

```text
yesterday + session date 8 May 2023
-> 7 May 2023
```

For duration expressions, it preserves duration form:

```text
since 2016 -> Since 2016
for four years -> four years
```

### Answer Override Policy

The resolver first lets the normal answer model generate an answer. Then:

```text
if the answer is empty, uncertain, or says "Not mentioned":
    use the deterministic temporal candidate
elif the old normalizer did not change the answer and the candidate is benchmark-style:
    use the deterministic temporal candidate
else:
    keep the normalized model answer
```

This is meant to avoid over-overriding already-good answers while rescuing
cases where evidence is available but the answer model fails to verbalize time.

### Diagnostics Added

Each resolved answer stores:

```text
temporal_resolver_used
temporal_selected
temporal_candidate_count
temporal_initial_response
temporal_normalized_response
temporal_final_response
```

These fields should be inspected together with:

```text
retrieval_diagnostics.evidence_hit_all
metrics.llm_judge_score
```

### Evaluation Plan

Run a focused Category 2 experiment first:

```text
--categories 2
```

Compare against v13:

```text
category 2 F1
category 2 EM
category 2 LLM judge score
all_hit_but_wrong count
empty / Not-mentioned temporal answers
temporal resolver used count
```

If the focused run improves Category 2 without obvious regressions in examples,
the next version should replace the temporary `category == 2` gate with
question-type-aware temporal routing.

---

## 15. v15 Diagnosis: Evidence Bundle Completeness and Answer Realization

### Motivation

The v13 evidence-first run and the v14 temporal-resolver run expose a broader
failure mode than any single benchmark category.

v13 on categories 1, 2, 4, and 5:

```text
total questions: 476
overall F1: 0.3715
overall LLM judge score: 0.3824

category 1: hit_any 65/74, hit_all 25/74, judge 19/74
category 2: hit_any 77/90, hit_all 75/90, judge 54/90
category 4: hit_any 120/200, hit_all 118/200, judge 103/200
category 5: hit_any 80/112, hit_all 78/112, judge 6/112
```

v14 focused on category 2:

```text
category 2 hit_any: 78/90
category 2 hit_all: 76/90
category 2 F1: 0.6782
category 2 LLM judge score: 0.6444
```

The temporal resolver improved category 2 answer realization, but it barely
changed retrieval coverage. Therefore the current bottleneck has three layers:

1. the system often retrieves at least one relevant evidence item, but not the
   complete evidence bundle;
2. when the complete evidence is retrieved, the answer model still omits answer
   qualifiers, list items, temporal granularity, or weak-inference wording;
3. lexical F1 can look acceptable while the semantic answer is still wrong.

### Key Diagnosis

The current implementation is reasonable as an evidence-preserving graph
retrieval prototype:

```text
raw LoCoMo turns are preserved
domain routing is now soft instead of filtering away evidence
graph edges expose same_storyline / similar_event / image_text_pair relations
answer prompts use question-type instructions
temporal normalization is treated as a support component
```

However, the current ranking objective is still mostly item-level relevance.
It scores each memory independently, then appends limited graph or local
neighbors. This does not directly optimize for answer coverage.

The strongest evidence is the gap between hit_any and hit_all:

```text
category 1: 87.8% hit_any vs 33.8% hit_all
category 2: 85.6% hit_any vs 83.3% hit_all
category 4: 60.0% hit_any vs 59.0% hit_all
category 5: 71.4% hit_any vs 69.6% hit_all
```

Category 1 makes the multi-evidence weakness visible, but the underlying issue
is not category-specific. The retriever lacks a general objective for selecting
a compact set of complementary evidence blocks.

### Additional Problems Exposed

#### 1. Category-specific gates are accumulating

The current code contains several benchmark-label gates:

```python
if category_int == 1: ...
elif category_int == 4: ...

if category == 1:
    refine_cat1_answer_with_evidence(...)
```

These gates are useful for diagnostics, but they should not become the final
paper design. The final version should route by inferred evidence requirement:

```text
question -> evidence need profile -> retrieval bundle policy -> answer policy
```

The profile can include:

```text
single_span
multi_item_list
temporal
weak_inference
comparison_or_preference
yes_no_fact
```

This keeps the method benchmark-independent and preserves the paper's main
storyline.

#### 2. Category 5 is mainly an answer-realization failure

Category 5 has relatively high gold-evidence coverage:

```text
hit_all: 78/112
judge correct: 6/112
not-mentioned style predictions: 100/112
```

This means the model is treating weak-inference questions as unsupported
factual-span questions. The fix should not be a category 5 module. It should be
an answer policy for inferred weak-inference questions:

```text
if evidence supports a tendency, return "Likely yes/no" or the most likely
preference/trait with a short evidence anchor.
```

This belongs under evidence-grounded answer realization, not under a new
retrieval mechanism.

#### 3. F1 is not reliable enough for iteration decisions

Examples show high lexical overlap but semantic failure:

```text
prediction: counseling and mental health
gold: counseling or mental health for Transgender people

prediction: nature
gold: dinosaurs, nature

prediction: The week before 17 July 2023
gold: The weekend before 17 July 2023
```

Future diagnostics should report:

```text
hit_all_but_wrong
miss_all_but_correct
high_f1_judge_wrong
not_mentioned_rate
empty_answer_rate
answer_type_distribution
```

LLM judge is currently more informative than F1 for semantic correctness, but
F1 is still useful for rough regression tracking.

#### 4. Temporal resolution needs question-aware candidate selection

v14 improved category 2, but remaining errors show candidate selection can pick
the wrong temporal expression from a retrieved bundle:

```text
charity race: Saturday vs Sunday
pride parade: wrong retrieved temporal event selected
week vs weekend granularity mismatch
last year should sometimes normalize to the anchored year
```

The resolver should become question-type-aware rather than category-aware, and
it should choose the temporal candidate whose evidence text best matches the
event phrase in the question.

#### 5. Graph edges help, but edge construction is still too cue-list driven

The same_storyline graph is a useful innovation, but many cues are currently
hand-curated. This is acceptable as a first prototype, but the paper story is
stronger if v15 frames this as:

```text
typed evidence graph edges are used to construct complementary bundles
```

instead of:

```text
typed edge rules are individually tuned for many topics
```

The next implementation should use the existing edge types, but change the
selection objective from "top related items" to "minimal sufficient evidence
bundle".

### v15 Repair Direction

Do not add a new mechanism. Repair the existing five innovation points as one
pipeline:

```text
Evidence-preserving writer
  -> typed episodic graph
  -> soft domain / hybrid candidate pool
  -> evidence bundle coverage selector
  -> question-type answer realization
  -> bundle-level reflection diagnostics
```

The key addition is a general evidence need profile:

```text
question text
  -> answer type
  -> required evidence cardinality
  -> required evidence facets
```

Example profiles:

```text
multi_item_list:
  select diverse blocks covering distinct answer items

temporal:
  select event-matching blocks with temporal expressions and session anchors

weak_inference:
  select positive/negative support blocks and require likely-style answer

single_span:
  select the strongest direct evidence and one local clarifier
```

This can replace category-specific gates while preserving the existing
innovations:

```text
category-specific Cat1 coverage ranking
  -> profile-based multi_item_list coverage ranking

category-specific Cat2 temporal resolver
  -> profile-based temporal answer realization

category 5 prompt weakness
  -> profile-based weak_inference answer realization
```

### Recommended Next Experiment

The next version should not target one category. It should add diagnostics and a
small profile layer, then run the same mixed category setting as v13.

Implementation sketch:

1. rename category-specific answer planning into evidence profile planning;
2. keep current retrieval sources, but select the final context by coverage
   facets rather than pure item score;
3. expose structured evidence blocks to the answer model for all profiles;
4. route temporal normalization by profile, not benchmark category;
5. log hit_all_but_wrong, high_f1_judge_wrong, not_mentioned_rate, and selected
   evidence facets.

Expected outcome:

```text
hit_all should improve for multi-item questions
not-mentioned rate should drop for weak-inference questions
high_f1_judge_wrong should become easier to diagnose
category 2 gains from v14 should remain without using category labels
```

---

## 16. v15 Implementation Step 1-3: Evidence Profile Routing

### Scope

This iteration implements only the low-risk part of the v15 plan:

1. add a question-text evidence need profile;
2. keep answer type and evidence profile as separate decisions;
3. generalize the old Category 1 coverage ranking into a `multi_item_list`
   profile path.

It does not yet implement:

```text
temporal candidate reranking by event phrase
weak-inference answer repair beyond profile-level prompt instruction
new graph edge types
new retrieval source modules
```

### Code Changes

In `test_advanced_robust.py`:

```text
infer_answer_type(question)
  -> controls output format

infer_evidence_need_profile(question, answer_type)
  -> controls evidence bundle policy
```

The first profile set is:

```text
single_span
multi_item_list
temporal
weak_inference
yes_no_fact
```

The answer path now performs retrieval after profile inference:

```python
answer_type = infer_answer_type(question)
evidence_profile = infer_evidence_need_profile(question, answer_type)
raw_context = retrieve_memory(..., evidence_profile=evidence_profile)
```

For temporal questions, the v14 temporal resolver is retained as a generic
question-type component:

```python
if answer_type == "temporal":
    response = resolve_temporal_answer(response, raw_context, question)
```

For multi-item questions, the old Category 1 evidence refinement is now:

```python
if evidence_profile == "multi_item_list":
    response = refine_multi_item_answer_with_evidence(...)
```

In `memory_layer_robust.py`, retrieval accepts the same profile:

```python
find_related_memories_raw(..., evidence_profile=evidence_profile)
```

The old Category 1 coverage selector is generalized:

```text
_category1_query_expansion      -> _multi_item_query_expansion
_category1_slot_profile         -> _multi_item_slot_profile
_category1_slot_signal          -> _multi_item_slot_signal
_select_category1_coverage_ranked -> _select_multi_item_coverage_ranked
```

New diagnostics use `multi_item_*` fields instead of `cat1_*` fields.

### Why This Should Be a Clean Test

This patch changes the trigger condition, not the core multi-evidence
algorithm. Therefore a micro experiment can answer one narrow question:

```text
Does question-text profile routing preserve the useful Category 1 behavior
while making the method less benchmark-label-dependent?
```

For a Category 2-only v15 run, the expected behavior is:

```text
temporal resolver still activates through answer_type == temporal
no category == 2 gate is used
retrieval uses the default non-category-2 graph expansion policy
```

The most important diagnostics to inspect are:

```text
answer_diagnostics.answer_type
answer_diagnostics.evidence_profile
answer_diagnostics.temporal_resolver_used
retrieval_diagnostics.evidence_hit_all
metrics.llm_judge_score
```

### Follow-up Todo Tracking

Open follow-up items are tracked in:

```text
a_mem_locomo_todo.md
```

The current deferred research item is summary/event nodes for graph support.
This should be treated as an auxiliary routing idea, not as a replacement for
raw evidence nodes and not as a reason to expand the persistent edge taxonomy.

---

## 17. v16 Repair: Profile Routing, Temporal Matching, and Short Evidence Prompt

### Motivation

The first v15 Cat1 micro-run showed that prompt shortening alone is not enough.
The prompt became shorter, but many Cat1 questions were routed to `single_span`,
which reduced evidence coverage:

```text
v13 Cat1 hit_all: 25/74
v15 Cat1 hit_all: 17/74
v13 Cat1 judge: 19/74
v15 Cat1 judge: 9/74
```

The failure was mainly profile routing and evidence bundle exposure, not a need
for more graph edge types.

### Implemented Fixes

The v16 repair implements the five active todos:

1. temporal resolver now scores temporal candidates by event match instead of
   choosing the first candidate;
2. weak-inference instructions now explicitly prefer a likely conclusion when
   retrieved evidence supports one;
3. `infer_answer_type` no longer treats every `when` occurrence as temporal;
4. `infer_evidence_need_profile` routes more multi-answer and attribute-bundle
   questions into `multi_item_list`;
5. the answer LLM now receives a compact structured evidence list instead of
   the full raw retrieved context.

### Short Evidence List

The final answer prompt now uses:

```text
[Evidence i]
dia_id:
session_date:
speaker:
relation:
fact:
image_caption:
image_query:
matched_question_terms:
```

The raw context is still returned for retrieval diagnostics and post-processing.
This keeps `hit_any` / `hit_all` comparable while reducing the answer prompt.

### Scope Note

This repair does not add graph edge types and does not implement summary/event
nodes. The summary/event-node idea remains deferred in `a_mem_locomo_todo.md`.

---

## 18. v17 Restore: v13 Baseline with Conservative Answer Guards

### Motivation

The v16-style profile routing and short evidence prompt were too broad for the
next stable experiment. The Cat1 micro-run regressed sharply, while Cat2 only
improved slightly. The next step therefore restores the code path to the
pre-v15/v16 stable baseline, then applies only two conservative answer-side
guards:

```text
restore stable v13-style retrieval / answer pipeline
  -> strengthen Not mentioned instruction
  -> prevent Cat1 evidence rerank from erasing a supported initial answer
```

This keeps the experiment focused on whether the high `Not mentioned` rate is
mainly an answer-realization problem rather than a retrieval/profile rewrite
problem.

### Restored Scope

The active code is restored from the stable pre-v15/v16 line identified in git
history. This means the following v16 changes are not active in v17:

```text
general evidence_profile routing
short evidence list for every category
event-matched general temporal resolver
multi-item profile rename / routing rewrite
```

The old Cat1 evidence rerank and old Cat2 temporal resolver remain, matching the
stable baseline more closely.

### v17 Additions

The main answer prompt now says:

```text
partial evidence -> answer the supported part
list questions -> include all supported items
weak judgment / likely questions -> give the best supported conclusion
Not mentioned -> only when no retrieved block mentions the needed subject/event
```

The Cat1 evidence rerank prompt now also forbids empty answers and forbids
returning `Not mentioned in the conversation` when structured evidence mentions
the requested subject/event/item.

### Rerank Safety Guard

After the Cat1 rerank LLM returns, v17 rejects outputs that are only labels,
empty strings, or `Not mentioned` replacements when:

```text
structured evidence exists
and the initial answer was non-empty
```

In those cases the system falls back to the initial answer and records:

```text
answer_diagnostics.cat1_rerank_fallback_to_initial = true
```

### Version Tag

Use this experiment label for result files:

```text
v17_v13_prompt_guard
```

---

## 19. v18 Rewrite Memory: Retrieval-Oriented Evidence Normalization

### Motivation

The v17 Cat1/Cat2 run showed that the remaining bottleneck is retrieval
ranking, not answer refusal. Cat1 especially has high `hit_any` but low
`hit_all`, which means the system often finds part of the answer evidence but
does not rank the full evidence set into the final context.

The suspected cause is that raw dialogue turns are too weak as retrieval
objects:

```text
pronouns and ellipsis
relative time expressions
chatty filler words
answer facts split across dialogue wording
image facts separated from text facts
```

### Implemented Change

v18 adds `rewrite_content` to every `RobustMemoryNote`.

The rewrite is an LLM-generated, self-contained evidence sentence. It resolves
pronouns, preserves names/activities/objects/places/dates, anchors relative
time when possible, and keeps image facts when they are answer-bearing.

No `rewrite_summary` field is added. The single retrieval representation is:

```text
rewrite_content
keywords
tags
visual cue: image_caption + image_query
metadata: dia_id + session_date + speaker
```

### Where Rewrite Is Used

The following stages now use `rewrite_content` instead of raw dialogue content:

```text
embedding index text
BM25 document text
lexical relevance
exact/entity overlap ranking
Cat1 slot-token coverage rerank
offline domain annotation brief
offline graph event text
typed graph edge profile
write-time same-entity/topic retrieval
```

Graph edge types are unchanged:

```text
same_storyline
similar_event
same_character
image_text_pair
```

This keeps the innovation on memory representation quality rather than adding a
larger edge taxonomy.

### Final Answer Context

The final answer still uses the original raw memory sentence for evidence
faithfulness and compatibility with existing LoCoMo diagnostics. The final
context size is reduced to top 10 evidence blocks:

```text
DEFAULT_FINAL_BUNDLE_MAX_SIZE = 10
DEFAULT_CAT1_PRIMARY_BUNDLE_SIZE = 10
DEFAULT_CAT1_MAX_CONTEXT_BLOCKS = 10
```

This tests whether rewrite-based ranking can move the right raw evidence into a
smaller final prompt, instead of relying on 16-block Cat1 admission.

### Cache Note

The retriever and domain-graph cache versions are bumped:

```text
robust_retrieval_v7_rewrite_memory_index
domain_graph_v7_rewrite_memory_edges
```

Old pickled memories are backfilled with `rewrite_content` on load, then saved
again after graph preparation.

### Version Tag

Use this experiment label for result files:

```text
v18_rewrite_memory_top10
```

---

## 20. v19 Single Rewrite Index and Score Diagnostics

### Motivation

v18 used `rewrite_content` as the retrieval representation, but
`_memory_to_index_text` still repeated the main evidence text three times. That
was an old field-weighting heuristic from the raw-dialogue retrieval setting.
With normalized rewrite memory, the repetition is harder to justify and may
over-amplify broad terms such as family, support, event, and activity.

v19 removes this repetition and adds richer diagnostics so failed gold evidence
can be traced to the scoring mechanism that suppressed it.

### Index Text Change

The retriever document now contains one copy of the normalized memory:

```text
rewrite_content
metadata: dia_id + session_date + speaker + keywords + tags
status
visual cue: image_caption + image_query
```

The retriever cache version is bumped:

```text
robust_retrieval_v8_rewrite_single_index_debug
```

### Candidate Score Diagnostics

Each candidate debug entry now records:

```text
score_inputs
score_weights
score_contributions
combined_rank
source_ranks
```

`score_contributions` decomposes the final combined score into weighted terms:

```text
domain_embedding
domain_bm25
domain_lexical
global_embedding
global_bm25
global_entity
graph_expansion
domain_match
lexical
reliability
citation
session_bonus
```

`source_ranks` shows where the same memory ranked under individual scoring
channels, making it easier to tell whether a missed gold evidence item was
suppressed by dense retrieval, BM25, lexical overlap, entity matching, graph
expansion, or Cat1 coverage selection.

### Result File Diagnostics

The saved `candidate_debug` list is expanded from top 30 to top 100 candidates
per question. This is diagnostic-only: it does not increase the final evidence
context shown to the answer LLM.

### Version Tag

Use this experiment label for result files:

```text
v19_single_rewrite_index_debug
```
