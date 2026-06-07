# Domain-guided Episodic Memory Graph for LoCoMo

## 0. Current Consensus

This note records the current innovation design for improving A-Mem on LoCoMo-style long-term conversational memory QA.

The main goal is not to build a generic memory graph. The goal is to build a domain-guided episodic evidence retrieval system that can retrieve a small, precise memory bundle from a long two-person conversation.

Current first-version assumptions:

1. Build one domain tree for each independent long conversation sample.
2. Use a three-level domain tree.
3. Annotate each raw leaf memory with at most three domain paths.
4. Use domain routing as the main retrieval entry point.
5. Add only a small global fallback to reduce router failure.
6. Build graph edges only between leaf memories.
7. Do not create summary or generalized memories in the first version.
8. Do not persist update/conflict edges. Update or conflict should trigger memory rewriting or state adjustment.
9. Update retrieval behavior after each session, subject to later empirical tuning.
10. Implement the first version in `memory_layer_robust.py` directly, while keeping behavior easy to ablate.
11. Evaluate the first implementation on LoCoMo categories 1, 2, and 4 first.
12. Use offline construction first. Online/session-incremental memory updates should be attempted only after offline retrieval shows clear gains.
13. Use an LLM to construct a domain tree for each sample and annotate each memory with domain paths.
14. Cache generated domain trees, memory-domain annotations, and graph edges locally so later runs can reuse them.

LoCoMo session definition:

```text
A session is one timestamped conversation segment inside a long conversation sample.
In the local dataset it appears as session_1, session_2, ..., each with a
session_N_date_time field. Each session contains about 10-47 dialogue turns.
```

---

## 1. Core Innovation

Proposed name:

```text
Domain-guided Episodic Memory Graph
```

Main idea:

```text
LoCoMo conversation
  -> per-conversation domain tree construction
  -> leaf memory domain annotation
  -> domain-routed seed retrieval
  -> leaf-level graph bundle expansion
  -> lifecycle/utility-aware reranking
  -> small evidence bundle for QA
```

The contribution should be framed as an evidence-first extension of A-Mem:

```text
Instead of retrieving from a flat memory pool or relying only on semantic neighbors,
the system first routes the query into a conversation-specific domain tree, retrieves
candidate evidence memories from selected leaf domains, expands them with lightweight
episodic graph edges, and reranks the final bundle using relevance, graph structure,
freshness, reliability, and historical usefulness.
```

---

## 2. LoCoMo-specific Memory Unit

The first version treats each dialogue turn as a raw leaf memory.

Raw leaf memories should not be rewritten into summaries in version 1.

Suggested fields:

```yaml
memory_id: locomo_sample0_D1_3
sample_id: sample_0
session_id: 1
dia_id: D1:3
session_date: 2023-05-08
speaker: Caroline
content: "I went to the LGBTQ support group yesterday."
image_caption: null
image_query: null
memory_level: leaf
domain_paths:
  - Personal Life / Identity / LGBTQ Support
entities:
  - Caroline
  - LGBTQ support group
temporal_expressions:
  - text: yesterday
    anchor: session_date
    normalized_date: 2023-05-07
status: active
reliability_alpha: 1.0
reliability_beta: 1.0
citation_count: 0
successful_citation_count: 0
failed_citation_count: 0
```

---

## 3. Domain Tree Construction

### 3.1 Scope

Build one domain tree for each independent LoCoMo long conversation sample.

Rationale:

```text
Each LoCoMo sample describes a different two-person relationship, life trajectory,
set of recurring topics, and temporal progression. A shared global tree may be too
coarse and may mix unrelated characters and events.
```

### 3.2 Tree Depth

Use three levels:

```text
Level 1: broad life area
Level 2: subtopic
Level 3: specific recurring theme or event type
```

Example:

```text
Personal Life / Family / Adoption
Personal Life / Identity / LGBTQ Support
Career Education / Mental Health / Counseling
Social Relationship / Friends / Melanie
Hobbies Interests / Reading / Books
Events Activities / Meetings / Support Group
```

### 3.3 Construction Method

Use an LLM before QA to construct a tree from the full conversation sample.

Input:

```text
all sessions
all dialogue turns
optional session summaries
speaker names
```

Output:

```json
{
  "sample_id": "sample_0",
  "domain_tree": [
    {
      "path": "Personal Life / Family / Adoption",
      "description": "Memories about adoption, family planning, parenting hopes, agencies, and related support.",
      "keywords": ["adoption", "family", "agency", "parenting"]
    }
  ]
}
```

### 3.4 Memory-domain Annotation

Each leaf memory can be assigned to at most three domain paths.

Rules:

```text
1. Prefer the most specific matching domain path.
2. Allow multiple domains only when the same turn genuinely touches multiple recurring topics.
3. Do not annotate with vague domains if a specific domain fits.
4. Keep domain paths inside the conversation-specific tree.
```

---

## 4. Domain-aware Seed Retrieval

### 4.1 Motivation

Full-pool retrieval may be expensive and noisy. Domain tree routing is intended to make retrieval purposeful and faster.

### 4.2 Retrieval Flow

```text
query
  -> route query to top-k domains
  -> retrieve seeds inside selected domain sublibraries
     - embedding retrieval
     - BM25 retrieval
     - entity/speaker matching
  -> add small global fallback
  -> deduplicate candidates
```

Suggested first-version parameters:

```yaml
domain_top_k: 3
domain_seed_top_k: 20
global_fallback_top_k: 5
final_bundle_size: 6
final_bundle_max_size: 8
```

These are first-version defaults. They should be tuned later based on retrieval recall,
runtime, noise ratio, and final QA performance.

### 4.3 Global Fallback

The global fallback is intentionally small.

Purpose:

```text
Prevent early domain-router mistakes from completely blocking gold evidence.
```

It should not dominate retrieval.

---

## 5. Leaf-level Graph Edges

Edges exist only between leaf memories.

A leaf memory can have multiple edges.

### 5.1 Retrieval Edges

Use only three retrieval edge types in version 1:

```text
temporal_anchor
similar_event
same_character
```

#### temporal_anchor

Definition:

```text
A memory contains a temporal expression or an anchor date useful for temporal QA.
```

This can be implemented either as:

```text
memory -> normalized temporal metadata
```

or as:

```text
memory_i -> memory_j
```

when two memories describe temporally connected events.

#### similar_event

Definition:

```text
Two leaf memories refer to the same or strongly related event, activity, plan, or recurring topic.
```

Likely cues:

```text
same session and nearby turns
shared entities
shared actions
shared event nouns
high lexical overlap
embedding similarity
```

This is expected to be the most important edge type for LoCoMo QA.

Version 1 construction:

```text
Use rules + embedding similarity.
Do not require an LLM call for every candidate edge in the first implementation.
```

Future stronger version:

```text
Use an LLM event-equivalence judge for high-value candidate pairs after rule/embedding
pre-filtering. This may improve quality, but should be added only after the cheaper
version is measured.
```

Cross-session similar_event is allowed.

Rationale:

```text
LoCoMo questions often require connecting evidence across multiple timestamped
conversation sessions. For example, a character may first mention researching
adoption agencies, later apply to agencies, and later attend an adoption-related
meeting. These turns are in different sessions but belong to one long-term event
chain.
```

Suggested first-version rules:

```text
same-session similar_event:
  same session + nearby turns + shared entity/action/topic

cross-session similar_event:
  different sessions + same_character + at least one shared domain path
  + shared core entity/event/action term
```

Do not require a special temporal-reasoning proof for cross-session similar_event in version 1.
The phrase "explainable time gap" only means that the two memories can reasonably be part
of the same long-term storyline after sorting by session date. In implementation terms,
the first version can approximate this with:

```text
session dates are known
and the newer memory does not contradict the older memory
and the shared event/domain still makes sense as a continuing topic
```

Initial edge weights:

```yaml
same_session_similar_event_weight: 1.0
cross_session_similar_event_weight: 0.6
```

#### same_character

Definition:

```text
Two memories involve the same core speaker or character.
```

This edge should have lower expansion weight because LoCoMo conversations can contain many memories from the same speaker.

Initial edge weight:

```yaml
same_character_weight: 0.4
```

### 5.2 Evidence Edges

Use only two evidence edge types:

```text
supports
derived_from
```

In version 1, summary memories are not created, so these edges may be logged for future use but do not need to drive the main retrieval pipeline.

#### supports

Definition:

```text
A raw leaf memory supports an answer, event, or future derived memory.
```

#### derived_from

Definition:

```text
A future derived or generalized memory is grounded in one or more raw leaf memories.
```

### 5.3 No Lifecycle Edges

Do not create persistent update/conflict edges.

Policy:

```text
If update or conflict is detected, trigger memory rewriting or state adjustment.
Do not add update/conflict as graph edges for bundle expansion.
```

---

## 6. Graph-based Bundle Expansion

Seed memories are expanded through selected graph edges.

Question-type-aware policy can be added later, but the first version should remain simple:

```text
temporal questions:
  prefer temporal_anchor + similar_event

factual questions:
  prefer similar_event

character/person questions:
  allow same_character but with strict cap

inference questions:
  use similar_event first, then same_character as weak support
```

Suggested first-version limits:

```yaml
expand_per_seed: 1-2
max_same_character_expansions: 1-2
max_graph_candidates_before_rerank: 40
```

---

## 7. Lifecycle-aware Bundle Reranking

The final bundle should be selected by utility, not only relevance.

Suggested utility signals:

```text
relevance
domain score
graph importance
freshness
reliability
citation count
state penalty
noise penalty
```

Suggested first-version formula:

```text
score(m, q) =
  w_rel * relevance(q, m)
+ w_domain * domain_score(q, m)
+ w_graph * graph_importance(m)
+ w_reliability * reliability(m)
+ w_citation * log(1 + citation_count(m))
+ w_fresh * freshness(m)
- w_state * state_penalty(m)
- w_noise * noise_penalty(m, q)
```

Reliability should not be raw citation count.

Use a Bayesian-style estimate:

```text
reliability(m) =
  (successful_citation_count + alpha)
  /
  (successful_citation_count + failed_citation_count + alpha + beta)
```

Why:

```text
A memory that is frequently retrieved but often hurts answers should not become more important.
```

---

## 8. Memory Update Mechanism

There are two update paths.

### 8.1 Periodic Session-level Update

After each session, update retrieval statistics and memory states.

Candidate statistics:

```text
answer correctness
retrieved memory IDs
cited memory IDs
successful citation count
failed citation count
domain hit/miss
missed gold evidence
noise memories
edge usefulness
```

Potential actions:

```text
increase reliability for useful memories
decrease reliability for noisy memories
adjust edge weights
adjust domain priors
mark stale or low-confidence memories
```

### 8.2 Recall-triggered Update

When a query retrieves memories that indicate possible update or conflict:

```text
1. Ask whether the memories describe the same subject/event.
2. Ask whether newer evidence updates older evidence.
3. If yes, rewrite the affected non-raw memory or adjust state.
4. Preserve raw leaf memories.
5. Do not create update/conflict edges.
```

First version:

```text
Since summary memories are not used, recall-triggered update can focus on reliability,
state, and edge/domain weights rather than rewriting raw content.
```

---

## 9. Evaluation Plan

### 9.1 QA Metrics

```text
Accuracy
F1
ROUGE
semantic similarity
```

### 9.2 Retrieval Metrics

```text
Evidence Recall@K
Gold Evidence Rank
Answerable Context Rate
Noise Ratio
```

### 9.3 Domain Tree Metrics

```text
Domain Routing Hit Rate
Domain Recall@top-k
Domain Search Cost Reduction
Routing Confidence vs Answer Accuracy
```

### 9.4 Memory Utility Metrics

```text
citation_count
successful_citation_count
failed_citation_count
Bayesian reliability
edge usefulness
domain usefulness
```

---

## 10. First-version Ablation

Recommended ablations:

```text
B0: current memory_layer_robust.py
B1: per-conversation domain tree + domain annotation
B2: domain-routed retrieval without graph expansion
B3: + small global fallback
B4: + leaf-level graph expansion
B5: + lifecycle/utility reranking
B6: + session-level feedback update
```

Important comparisons:

```text
domain-only retrieval vs domain retrieval + global fallback
embedding-only seed retrieval vs embedding + BM25 + entity
without graph expansion vs with temporal_anchor/similar_event/same_character
relevance-only rerank vs lifecycle/utility rerank
```

Execution plan:

```text
1. Modify memory_layer_robust.py directly.
2. Run local tests after implementation.
3. Run a focused LoCoMo evaluation on categories 1/2/4.
4. Commit and push the tested changes to git@github.com:moyutop1/A-mem-baseline.git.
```

Preprocessing/cache decision:

```text
For each LoCoMo sample, the first offline run may call the LLM to:
1. build the three-level domain tree;
2. annotate each leaf memory with up to three domain paths;
3. construct initial leaf-level graph edges.

The generated structures should be cached locally. Later QA/evaluation runs should
reuse the cache whenever possible to avoid repeated LLM cost and reduce variance.
```

---

## 11. Open Questions

Still unresolved:

```text
1. Whether temporal_anchor should be stored as metadata or explicit memory-time edges.
2. How to define noise memory automatically.
3. Whether generated summaries should be added if version 1 gains are weak.
4. Exact LLM prompt/schema for domain tree construction and memory-domain annotation.
5. Exact thresholds for cross-session similar_event construction.
```

---

## 12. First Ratio-0.1 DeepSeek Diagnosis

Result file:

```text
robust_domain_graph_locomo10_cat124_ratio01_deepseek.json
```

Setup:

```text
model: deepseek-chat
categories: 1, 2, 4
ratio: 0.1
questions: 139
compress_context: false
```

Observed QA performance:

```text
overall exact_match: 0.173
overall F1: 0.375

category 1 exact_match: 0.000
category 1 F1: 0.159

category 2 exact_match: 0.405
category 2 F1: 0.565

category 4 exact_match: 0.129
category 4 F1: 0.373
```

Evidence recall after normalizing gold evidence IDs:

```text
category 1 Evidence Recall@bundle: 15 / 32 = 0.469
category 2 Evidence Recall@bundle: 25 / 37 = 0.676
category 4 Evidence Recall@bundle: 36 / 70 = 0.514
overall Evidence Recall@bundle: 76 / 139 = 0.547
```

Main diagnosis:

```text
1. Category 1 multi-hop performance is bottlenecked by incomplete evidence recall.
   Many questions need multiple memories, but the current bundle often retrieves only
   one supporting memory or misses all gold turns.

2. Category 4 single-hop also misses too many exact evidence turns, so seed retrieval
   and domain routing still need stronger lexical/entity fallback.

3. Category 2 is the strongest category, but temporal answers still fail when the
   right memory is present but the model or normalization chooses the wrong date phrase.

4. Graph expansion is active, but local_context appears in almost every query and can
   dominate the final bundle when the primary seed is wrong.

5. similar_event helps, but current rule/embedding edges are still not enough for
   multi-session multi-hop aggregation.
```

Observed context pattern:

```text
average retrieved dia_ids per question: about 8.86
average raw_context length: about 5972 characters
local_context relation count: 684
similar_event relation count: 260
same_character relation count: 8
```

Implication:

```text
The current first version retrieves a fairly large bundle, but not always the right
bundle. The next iteration should improve seed precision and multi-evidence coverage
before adding heavier memory rewriting or summary memories.
```

Recommended next changes:

```text
1. Add evidence-aware global lexical/entity fallback larger than the current top-5,
   especially for exact action/object questions.

2. Reduce automatic local_context expansion. Only add adjacent turns when the neighbor
   has lexical/entity overlap with the query or the primary memory has high confidence.

3. Make category 1 retrieval multi-hop aware: gather candidates from multiple domains
   and enforce diversity across sessions/entities before reranking.

4. Add query-type aware retrieval policies:
   - category 1: prefer multi-seed, cross-session similar_event, entity/action coverage
   - category 2: prefer temporal_anchor and date-bearing memories
   - category 4: prefer exact BM25/entity match over broad embedding match

5. Add an optional compression/rerank step for category 1/4 after retrieval, because
   many hit-but-wrong examples contain the gold evidence but the answer selects only
   a partial or noisy fact.
```

Implemented next patch:

```text
Version: robust_retrieval_v4_stronger_fallback

1. Increase global fallback while keeping domain retrieval as the main path:
   - global embedding fallback: top 5
   - global BM25 fallback: top 15
   - global entity/action fallback: top 10

2. Add exact entity/action fallback scoring:
   - speaker/person overlap
   - query token overlap
   - exact action words such as research, apply, paint, read, camp, support,
     birthday, identity, beach, conference, meeting

3. Reduce local_context expansion:
   - radius 2 -> radius 1
   - only first two primary seeds may add local context
   - each primary seed may add at most one local neighbor
   - neighbor must have lexical overlap with the query

4. Bump retrieval and domain-graph cache versions so old caches are rebuilt.
```

Implemented v5 patch:

```text
Version: robust_retrieval_v5_source_aware_category_policy

1. Replace seed_indices set with a source-aware candidate score table.
   Each candidate now keeps:
   - domain_embedding
   - domain_bm25
   - domain_lexical
   - global_embedding
   - global_bm25
   - global_entity
   - domain_match
   - source_tags

2. Global fallback scores now participate in final reranking.
   This fixes the v4 issue where global fallback could add candidates but their
   BM25/entity strength was not preserved in the final score.

3. Add category-aware retrieval weights:
   - Category 1: balance domain and global evidence, with a small session diversity bonus.
   - Category 4: emphasize global_bm25 and global_entity; weaken broad embedding/domain signals.
   - Category 2: keep a balanced policy for now.

4. Limit graph expansion by category:
   - Category 4 skips similar_event and same_character expansion by default.
   - Other categories allow at most one similar_event expansion per primary seed.
   - similar_event target must have lexical overlap with the query.

5. Store source-aware retrieval diagnostics in memory_system.last_candidate_debug.
   These diagnostics are not injected into the prompt, but can be logged later.

6. Bump retrieval and domain graph cache versions so old v4 caches are rebuilt.
```

Implemented v6 diagnostics patch:

```text
Version behavior: retrieval policy unchanged from v5.

1. Add per-question retrieval_diagnostics to result JSON:
   - gold_evidence
   - retrieved_dia_ids
   - evidence_hit_any
   - evidence_hit_all
   - missed_gold_evidence
   - relation_counts
   - routed_domains
   - candidate_debug

2. Add top-level retrieval_diagnostics_summary by category.

3. Store last routed domains in the memory layer so routing decisions can be inspected.

Purpose:
Determine whether failures come from candidate recall, final bundle selection, graph
expansion noise, or answer generation. This should be done before enabling more
aggressive Cat1 slot coverage or evidence compression policies.
```

Implemented Cat1 coverage bundle-selection patch:

```text
Scope: Category 1 only. Category 2 and Category 4 retrieval policies are unchanged.
Cache behavior: retriever/domain graph cache versions are unchanged because this only
reranks already-built candidates at query time.

1. Add coverage-aware primary seed selection for Category 1:
   - Extract slot tokens from each candidate memory content, image fields, context,
     keywords, and tags.
   - Remove tokens already present in the query.
   - Greedily front-load candidates that add new slot tokens while retaining the
     existing retrieval score as the base score.
   - Add a small session-diversity bonus so multi-hop answers are less likely to be
     collapsed into one local conversation segment.

2. Protect primary evidence slots before graph expansion:
   - For Category 1, the first up to four selected primary memories enter the bundle
     before local/graph expansion can consume context budget.
   - Category 1 graph expansion prefers temporal_anchor/supports/derived_from before
     similar_event.
   - Category 1 expansion is capped at one graph neighbor per primary after the
     protected primary segment.

3. Add diagnostics in candidate_debug:
   - cat1_selected_primary
   - cat1_selected_rank
   - cat1_selection_score
   - cat1_coverage_gain
   - cat1_new_slot_tokens
   - cat1_slot_tokens
   - cat1_lexical_score

Purpose:
Test whether the weak Cat1 F1 is mainly caused by candidate evidence being available
but not surviving into the final answer bundle. This patch should first be evaluated
with --categories 1 only before adding evidence preservation or compression.
```

Implemented Cat1 slot-aware answer patch:

```text
Scope: Category 1 only. Category 2 and Category 4 remain unchanged.
Cache behavior: cache versions remain unchanged because this is query-time selection
and answer-time reranking only.

1. Replace broad Cat1 token coverage with answer-slot-aware coverage:
   - Infer a stable slot type from question wording:
     count, place, person, book, event, activity, item, status, or fact.
   - Score candidates with slot cues and target-term hits in addition to base
     retrieval score and weak token coverage.
   - Add diagnostics:
     cat1_answer_slot_type
     cat1_answer_slot_score
     cat1_slot_cue_hits
     cat1_slot_target_hits

2. Make Cat1 bundle construction primary-first:
   - Use up to final_bundle_max_size primary candidates for Category 1.
   - Disable local_context and graph expansion for Category 1 in this experiment.
   - Purpose: isolate whether Cat1 failures are caused by evidence selection rather
     than expansion noise.

3. Add Cat1 evidence-preserving answer rerank:
   - First generate the normal answer.
   - Then build a structured evidence list from the raw retrieved context.
   - Ask the model to revise the answer using only the structured evidence.
   - Preserve every distinct supported item for list-style questions.
   - Add answer_diagnostics with:
     cat1_evidence_rerank_used
     cat1_evidence_block_count
     cat1_initial_response
     cat1_refined_response
     cat1_evidence_preview

Expected effect:
This should mainly help hit-but-wrong and candidate-hit-but-context-missed Cat1
questions. It cannot solve questions where gold evidence is absent from the
candidate pool.
```
