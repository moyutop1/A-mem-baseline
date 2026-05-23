# memory_layer_robust.py interface guide

This document summarizes every class and function in `memory_layer_robust.py`.
It is intended as a compact interface map for reproducing A-Mem and designing extensions.

## File role

`memory_layer_robust.py` implements the robust version of the A-Mem memory layer.
Compared with `memory_layer.py`, this file avoids strict JSON schema calls and instead uses plain-text prompts plus parsers from `llm_text_parsers.py`.

High-level responsibilities:

- provide LLM backend wrappers for OpenAI, Ollama, SGLang, vLLM, and LiteLLM
- create memory notes with extracted metadata
- store memory notes in an in-memory dictionary
- retrieve related memories with embeddings
- use LLM calls to decide memory evolution
- create links between memories
- update old memories' context and tags

High-level write path:

```text
RobustAgenticMemorySystem.add_note(content)
  -> RobustMemoryNote(...)
       -> RobustMemoryNote.analyze_content(...)
  -> RobustAgenticMemorySystem.process_memory(note)
       -> find_related_memories(note.content, k=5)
       -> LLM evolution decision
       -> optional strengthen links
       -> optional update neighbor metadata
  -> store note in self.memories
  -> add note text/metadata to embedding retriever
```

High-level read path:

```text
find_related_memories_raw(query, k)
  -> embedding retriever top-k
  -> append each hit's linked neighbor memories
  -> return string context for QA
```

## Imports and external dependencies

Important imports:

| Import | Use |
|---|---|
| `SimpleEmbeddingRetriever` from `memory_layer.py` | Embedding-based memory retrieval. |
| `ANALYZE_CONTENT_PROMPT` | Prompt for extracting keywords/context/tags. |
| `EVOLUTION_DECISION_PROMPT` | Prompt for deciding whether memory evolution should happen. |
| `STRENGTHEN_DETAILS_PROMPT` | Prompt for choosing memory links and updated tags. |
| `UPDATE_NEIGHBORS_PROMPT` | Prompt for updating old memory context/tags. |
| `FOCUSED_KEYWORDS_PROMPT` | Fallback prompt if initial keyword parsing fails. |
| parser functions from `llm_text_parsers.py` | Convert plain-text LLM outputs into structured dictionaries/lists. |

Important note:

- Retrieval itself is not defined in this file. It uses `SimpleEmbeddingRetriever` imported from `memory_layer.py`.
- In this robust version, retrieval is embedding-only cosine similarity.

## Function: `retry_llm_call(max_retries: int = 2, base_delay: float = 1.0)`

Purpose:

- Decorator factory for retrying transient LLM call failures.
- Used on all concrete `get_completion` methods.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `max_retries` | int | `2` | Number of retries after the first failed attempt. Total attempts are `max_retries + 1`. |
| `base_delay` | float | `1.0` | Initial sleep time in seconds before retrying. Delay doubles after each failure. |

Returns:

- A decorator that wraps an LLM call function.

Behavior:

```text
call wrapped function
  -> if success, return result
  -> if failure and retries remain, sleep base_delay * 2^attempt
  -> if all attempts fail, log error and re-raise last exception
```

Side effects:

- Logs retry warnings and final failure errors.
- Sleeps between failed attempts.

Innovation hooks:

- Add retry filters so only network or rate-limit errors retry.
- Add jitter to avoid thundering herd failures.
- Expose retry settings through CLI or config.

## Class: `RobustBaseLLMController`

Purpose:

- Abstract base class for all robust LLM controllers.
- Defines a common plain-text completion interface with no JSON schema dependency.

Class attribute:

```python
SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."
```

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Abstract method that concrete LLM backends must implement.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `prompt` | string | required | User prompt sent to the LLM. |
| `temperature` | float | `0.7` | Sampling temperature. |

Returns:

- Plain-text LLM response.

### `check_connectivity(self)`

Purpose:

- Sends a simple test prompt to verify the backend is reachable.

Parameters:

- None.

Returns:

- No explicit return value.

Raises:

- `ConnectionError` if the backend returns an empty response or any call exception occurs.

Side effects:

- Calls the configured LLM once with temperature `0.0`.

## Class: `RobustOpenAIController`

Purpose:

- OpenAI chat-completions backend wrapper.

### `__init__(self, model: str = "gpt-4", api_key: Optional[str] = None)`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | string | `"gpt-4"` | OpenAI model name. |
| `api_key` | string or `None` | `None` | API key. If missing, reads `OPENAI_API_KEY` from environment. |

Raises:

- `ImportError` if `openai` package is missing.
- `ValueError` if no API key is available.

Side effects:

- Creates `OpenAI(api_key=api_key)` client.

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Calls OpenAI chat completions and returns message content.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `prompt` | string | required | Prompt text. |
| `temperature` | float | `0.7` | Sampling temperature. |

Returns:

- `response.choices[0].message.content`

Request settings:

- system message: `SYSTEM_MESSAGE`
- max tokens: `1000`
- no `response_format`

Side effects:

- External API call.
- Retried by `retry_llm_call(max_retries=2)`.

## Class: `RobustOllamaController`

Purpose:

- Direct Ollama Python-library backend wrapper.

### `__init__(self, model: str = "llama2")`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | string | `"llama2"` | Ollama model name, such as `qwen2.5:3b`. |

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Calls local Ollama chat API through the `ollama` Python package.

Returns:

- `response["message"]["content"]`

Raises:

- `ImportError` if `ollama` package is missing.

Side effects:

- Local Ollama call.
- Retried by `retry_llm_call(max_retries=2)`.

## Class: `RobustSGLangController`

Purpose:

- Direct SGLang HTTP backend wrapper.

### `__init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000)`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | string | `"llama2"` | Model name. Stored but not directly included in the request payload. |
| `sglang_host` | string | `"http://localhost"` | SGLang server host. |
| `sglang_port` | int | `30000` | SGLang server port. |

Internal fields:

- `self.base_url = f"{sglang_host}:{sglang_port}"`

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Sends a POST request to SGLang `/generate`.

Request endpoint:

```text
{base_url}/generate
```

Request payload:

```python
{
    "text": prompt,
    "sampling_params": {
        "temperature": temperature,
        "max_new_tokens": 1000,
    }
}
```

Returns:

- `response.json().get("text", "")` if HTTP status is 200.

Raises:

- `RuntimeError` on non-200 HTTP status.

Side effects:

- HTTP request to local or remote SGLang server.
- Retried by `retry_llm_call(max_retries=2)`.

## Class: `RobustVLLMController`

Purpose:

- vLLM OpenAI-compatible HTTP backend wrapper.

### `__init__(self, model: str = "llama2", vllm_host: str = "http://localhost", vllm_port: int = 30000)`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | string | `"llama2"` | vLLM-served model name. |
| `vllm_host` | string | `"http://localhost"` | vLLM server host. |
| `vllm_port` | int | `30000` | vLLM server port. |

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Sends a POST request to vLLM's OpenAI-compatible `/v1/chat/completions`.

Request endpoint:

```text
{base_url}/v1/chat/completions
```

Request payload:

```python
{
    "model": self.model,
    "messages": [
        {"role": "system", "content": self.SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ],
    "temperature": temperature,
    "max_tokens": 1000,
}
```

Returns:

- `response.json()["choices"][0]["message"]["content"]` if HTTP status is 200.

Raises:

- `RuntimeError` on non-200 HTTP status.

Side effects:

- HTTP request to local or remote vLLM server.
- Retried by `retry_llm_call(max_retries=2)`.

## Class: `RobustLiteLLMController`

Purpose:

- Generic LiteLLM backend wrapper.
- In the current factory, this class is defined but not selected by any backend branch.

### `__init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None)`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | string | required | LiteLLM model identifier. |
| `api_base` | string or `None` | `None` | Optional API base URL. |
| `api_key` | string or `None` | `None` | Optional API key. Defaults internally to `"EMPTY"`. |

Raises:

- `ImportError` if `litellm` package is missing.

### `get_completion(self, prompt: str, temperature: float = 0.7) -> str`

Purpose:

- Calls `litellm.completion(...)` with a common chat message format.

Returns:

- `response.choices[0].message.content`

Side effects:

- External or local LLM call depending on LiteLLM configuration.
- Retried by `retry_llm_call(max_retries=2)`.

Innovation hooks:

- Add a `backend == "litellm"` branch to `RobustLLMController` if you want to expose this controller.

## Class: `RobustLLMController`

Purpose:

- Factory class that selects one concrete robust LLM controller based on `backend`.
- The selected controller is stored as `self.llm`.

### `__init__(self, backend="sglang", model="gpt-4", api_key=None, api_base=None, sglang_host="http://localhost", sglang_port=30000, check_connection=False)`

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `backend` | literal string | `"sglang"` | One of `openai`, `ollama`, `sglang`, or `vllm`. |
| `model` | string | `"gpt-4"` | Model name passed to the selected controller. |
| `api_key` | string or `None` | `None` | Used by OpenAI controller. |
| `api_base` | string or `None` | `None` | Currently not used by any active branch. |
| `sglang_host` | string | `"http://localhost"` | Host for SGLang or vLLM-compatible local servers. |
| `sglang_port` | int | `30000` | Port for SGLang or vLLM-compatible local servers. |
| `check_connection` | bool | `False` | If true, calls `self.llm.check_connectivity()`. |

Returns:

- No explicit return value.

Internal selection:

```text
openai -> RobustOpenAIController
ollama -> RobustOllamaController
sglang -> RobustSGLangController
vllm -> RobustVLLMController
```

Raises:

- `ValueError` if backend is unsupported.

Important caveat:

- The constructor parameter names are SGLang-flavored, but they are also passed to vLLM as host and port.
- `api_base` is accepted but not used in current active branches.

Innovation hooks:

- Add backends here.
- Add cost tracking, request logging, or global rate limiting here.

## Class: `RobustMemoryNote`

Purpose:

- Represents one memory unit.
- Stores original content plus LLM-generated metadata and graph-style links.

Fields created:

| Field | Meaning |
|---|---|
| `content` | Original memory text. |
| `id` | UUID string unless provided. |
| `keywords` | Keywords extracted from content. |
| `links` | Neighbor memory indices selected during strengthen evolution. |
| `importance_score` | Currently defaults to `1.0`; not actively used in this file. |
| `retrieval_count` | Currently defaults to `0`; not actively updated in this file. |
| `timestamp` | Provided time or current time in `%Y%m%d%H%M`. |
| `last_accessed` | Provided time or current time in `%Y%m%d%H%M`. |
| `context` | Short semantic/context description. |
| `evolution_history` | Defaults to empty list; not actively updated in this file. |
| `category` | Defaults to `"Uncategorized"`. |
| `tags` | Broad tags extracted from content. |

### `__init__(self, content, id=None, keywords=None, links=None, importance_score=None, retrieval_count=None, timestamp=None, last_accessed=None, context=None, evolution_history=None, category=None, tags=None, llm_controller=None)`

Purpose:

- Creates a memory note.
- If `llm_controller` is provided and any of `keywords`, `context`, `category`, or `tags` is missing, it calls `analyze_content` to fill metadata.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `content` | string | required | Raw memory text. |
| `id` | string or `None` | `None` | Optional stable memory id. |
| `keywords` | list or `None` | `None` | Optional keywords. Generated if missing and controller exists. |
| `links` | dict/list or `None` | `None` | Optional linked memory indices. Defaults to empty list. |
| `importance_score` | float or `None` | `None` | Optional score. Defaults to `1.0`. |
| `retrieval_count` | int or `None` | `None` | Optional retrieval count. Defaults to `0`. |
| `timestamp` | string or `None` | `None` | Optional memory timestamp. |
| `last_accessed` | string or `None` | `None` | Optional last access timestamp. |
| `context` | string/list or `None` | `None` | Optional semantic context. Lists are joined into one string. |
| `evolution_history` | list or `None` | `None` | Optional evolution history. |
| `category` | string or `None` | `None` | Optional category. Defaults to `"Uncategorized"`. |
| `tags` | list or `None` | `None` | Optional tags. Generated if missing and controller exists. |
| `llm_controller` | `RobustLLMController` or `None` | `None` | Used to generate metadata. |

Returns:

- No explicit return value.

Side effects:

- May call the LLM once through `analyze_content`.

Important caveat:

- The condition checks whether `category` is missing, but `analyze_content` does not return category. The generated metadata only fills `keywords`, `context`, and `tags`.

### `analyze_content(content: str, llm_controller: RobustLLMController) -> Dict`

Purpose:

- Static method that extracts metadata from a memory's content.
- Uses plain-text LLM prompt and robust parsers.

Parameters:

| Parameter | Type | Meaning |
|---|---:|---|
| `content` | string | Raw memory text to analyze. |
| `llm_controller` | `RobustLLMController` | Controller whose `llm.get_completion` is called. |

Returns:

- Dictionary with at least:

```python
{
    "keywords": [...],
    "context": "...",
    "tags": [...]
}
```

Internal flow:

```text
ANALYZE_CONTENT_PROMPT.format(content=content)
  -> LLM get_completion
  -> parse_analyze_content(response, content)
  -> if keywords empty:
       FOCUSED_KEYWORDS_PROMPT
       -> LLM retry
       -> _parse_list_items
  -> validate_analysis_result
  -> return analysis
```

Failure behavior:

- If anything fails, logs an error and returns heuristic metadata:

```python
{
    "keywords": _heuristic_keywords(content),
    "context": _heuristic_context(content),
    "tags": _heuristic_keywords(content, 3),
}
```

Side effects:

- Usually calls LLM once.
- May call LLM twice if keyword extraction is empty.

Innovation hooks:

- Replace prompt-based extraction with structured local extraction.
- Add temporal/event/person metadata here.
- Add source fields such as speaker, session id, and evidence id.

## Class: `RobustAgenticMemorySystem`

Purpose:

- Core A-Mem memory manager.
- Handles memory storage, retrieval, memory linking, neighbor updates, and retriever consolidation.

Internal fields:

| Field | Meaning |
|---|---|
| `self.memories` | Dict from memory id to `RobustMemoryNote`. |
| `self.retriever` | `SimpleEmbeddingRetriever` for embedding search. |
| `self.llm_controller` | `RobustLLMController` used for metadata and evolution decisions. |
| `self.evo_cnt` | Number of times memory evolution returned true. |
| `self.evo_threshold` | How often to rebuild the retriever after evolution. |

### `__init__(self, model_name='all-MiniLM-L6-v2', llm_backend='sglang', llm_model='gpt-4o-mini', evo_threshold=100, api_key=None, api_base=None, sglang_host='http://localhost', sglang_port=30000, check_connection=False)`

Purpose:

- Initializes an empty memory system.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model_name` | string | `'all-MiniLM-L6-v2'` | SentenceTransformer model name for retrieval embeddings. |
| `llm_backend` | string | `'sglang'` | Backend for LLM calls. |
| `llm_model` | string | `'gpt-4o-mini'` | LLM model name. |
| `evo_threshold` | int | `100` | Rebuild retriever after this many successful evolutions. |
| `api_key` | string or `None` | `None` | Optional API key. |
| `api_base` | string or `None` | `None` | Accepted but not actively used by factory branches. |
| `sglang_host` | string | `'http://localhost'` | Local serving host. |
| `sglang_port` | int | `30000` | Local serving port. |
| `check_connection` | bool | `False` | Whether to send a startup test request. |

Side effects:

- Loads/initializes an embedding model through `SimpleEmbeddingRetriever`.
- Creates a robust LLM controller.

### `add_note(self, content: str, time: str = None, **kwargs) -> str`

Purpose:

- Main write API for adding a new memory.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `content` | string | required | Memory text to store. |
| `time` | string or `None` | `None` | Timestamp passed to `RobustMemoryNote(timestamp=time)`. |
| `**kwargs` | any | none | Forwarded to `RobustMemoryNote`, such as `keywords`, `context`, or `tags`. |

Returns:

- The new note's id string.

Internal flow:

```text
RobustMemoryNote(content, llm_controller, timestamp=time, **kwargs)
  -> process_memory(note)
  -> self.memories[note.id] = note
  -> retriever.add_documents([content + context + keywords + tags])
  -> if evolution happened, maybe consolidate_memories()
  -> return note.id
```

Side effects:

- May call LLM for note metadata.
- May call LLM for memory evolution.
- Mutates `self.memories`.
- Appends to embedding retriever.

Innovation hooks:

- Change what text gets indexed.
- Store structured memory metadata.
- Add memory importance or decay.
- Return richer info such as evolution decision and linked neighbors.

### `consolidate_memories(self)`

Purpose:

- Rebuilds the embedding retriever from current memory states.
- Useful because memory evolution can update old memories' context/tags, while old embeddings may still reflect earlier metadata.

Parameters:

- None.

Returns:

- No explicit return value.

Behavior:

```text
detect current retriever model name if possible
  -> create fresh SimpleEmbeddingRetriever
  -> for each memory:
       add memory.content + memory.context + keywords + tags to retriever
```

Side effects:

- Replaces `self.retriever`.
- Re-encodes all memories.

Innovation hooks:

- Replace full rebuild with local embedding update.
- Rebuild immediately after neighbor update.
- Add background or batched consolidation.

### `find_related_memories(self, query: str, k: int = 5) -> tuple`

Purpose:

- Finds existing memories related to a query using embedding search.
- Used during memory evolution, not final QA.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `query` | string | required | Query text. In `process_memory`, this is `note.content`. |
| `k` | int | `5` | Number of related memories to retrieve. |

Returns:

```python
(memory_str, indices)
```

Return fields:

| Field | Type | Meaning |
|---|---:|---|
| `memory_str` | string | Formatted text of retrieved memories for the evolution prompt. |
| `indices` | list/array | Indices into `list(self.memories.values())`. |

Behavior:

```text
if no memories:
  return "", []
else:
  indices = retriever.search(query, k)
  format each retrieved memory with index, timestamp, content, context, keywords, tags
```

Side effects:

- None, except embedding model work inside retriever search.

Important caveat:

- Returned indices are positional list indices, not stable memory ids.

Innovation hooks:

- Use `content + context + keywords + tags` for the new-memory query.
- Add BM25 or hybrid search.
- Add temporal filters.
- Return scores and ids.

### `find_related_memories_raw(self, query: str, k: int = 5) -> str`

Purpose:

- Finds memories related to a query and expands each hit with its linked neighbor memories.
- Used during QA to build answer context.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `query` | string | required | Query text, usually generated keywords from the question. |
| `k` | int | `5` | Number of directly retrieved memories. Also used as a loose cap for neighbor expansion. |

Returns:

- A string containing directly retrieved memories plus linked neighbor memories.

Behavior:

```text
indices = retriever.search(query, k)
for each index:
  append direct memory fields
  for each neighbor in memory.links:
    append neighbor memory fields
```

Side effects:

- None.

Important caveats:

- `links` are interpreted as positional indices into `all_memories`.
- No deduplication is performed, so the same memory can appear multiple times.
- Neighbor loop breaks when `j >= k`, which can include up to `k + 1` neighbors because `j` is checked after appending.
- No bounds check is done for neighbor indices.

Innovation hooks:

- Add deduplication.
- Use stable memory ids for links.
- Add hop limits and relation types.
- Add reranking or context compression after expansion.

### `process_memory(self, note: RobustMemoryNote) -> tuple`

Purpose:

- Core A-Mem evolution routine.
- Given a new note, retrieves related old memories and asks the LLM whether to evolve memory connections or update neighbors.

Parameters:

| Parameter | Type | Meaning |
|---|---:|---|
| `note` | `RobustMemoryNote` | Newly created memory note. |

Returns:

```python
(evo_label, note)
```

Return fields:

| Field | Type | Meaning |
|---|---:|---|
| `evo_label` | bool | Whether evolution happened successfully. |
| `note` | `RobustMemoryNote` | Possibly updated new note. |

Internal flow:

```text
find_related_memories(note.content, k=5)
  -> if no old memories, return False, note
  -> LLM call 1: EVOLUTION_DECISION_PROMPT
       parse_evolution_decision
       possible decisions:
         NO_EVOLUTION
         STRENGTHEN
         UPDATE_NEIGHBOR
         STRENGTHEN_AND_UPDATE
  -> if strengthen:
       LLM call 2: STRENGTHEN_DETAILS_PROMPT
       parse_strengthen_details
       note.links.extend(connections)
       optionally replace note.tags
  -> if update neighbor:
       LLM call 3: UPDATE_NEIGHBORS_PROMPT
       parse_update_neighbors
       update old memories' tags/context
  -> return True, note
```

Side effects:

- Calls LLM one to three times, depending on decision.
- Mutates the new note's `links` and `tags`.
- May mutate existing memories' `tags` and `context`.

Failure behavior:

- On any exception during evolution, logs error and returns `(False, note)`.
- The note can still be stored by `add_note`; it is just stored without evolution.

Important caveats:

- Candidate neighbor count is hard-coded to `k=5`.
- Link connections are stored as positional indices.
- Updating old memories does not immediately update their embeddings unless consolidation is triggered later.
- It returns `True` for any decision other than `NO_EVOLUTION`, even if the parsed strengthen/update outputs are empty.

Innovation hooks:

- This is the main place for A-Mem algorithmic innovation.
- Add typed edges, edge weights, relation explanations, bidirectional links, temporal constraints, conflict detection, or memory decay.
- Add a deterministic verifier after LLM evolution decisions.
- Add graph-aware neighbor expansion before LLM decision.

## Memory graph interpretation

The implementation behaves like a lightweight graph:

```text
node = RobustMemoryNote
edge = note.links entry
```

But it is not an explicit graph library.

Current graph representation:

```python
note.links = [0, 3, 5]
```

Meaning:

- The note is linked to memories at positions `0`, `3`, and `5` in `list(self.memories.values())`.

Limitations:

- links are index-based, not id-based
- no edge type
- no edge weight
- no edge timestamp
- no explanation
- no reverse edge by default

Recommended extension:

```python
{
    "target_id": "...",
    "relation": "same_event",
    "strength": 0.87,
    "reason": "Both memories mention the Yosemite hiking trip.",
    "created_at": "..."
}
```

## Retrieval behavior summary

When a new memory is added:

```text
query = new_note.content
documents = old_memory.content + old_memory.context + old_memory.keywords + old_memory.tags
retriever = SimpleEmbeddingRetriever
score = cosine_similarity(embedding(query), embedding(document))
top-5 results become candidate neighbors
```

When answering a question:

```text
query = generated question keywords
top-k memories = embedding search
context = top-k memories + each memory's linked neighbors
```

## API cost summary

Adding one memory can trigger:

```text
1 LLM call for metadata extraction
0-3 LLM calls for evolution:
  - decision
  - strengthen details
  - update neighbors
```

So one memory insertion can cost roughly:

```text
1 to 4 LLM calls
```

QA answering happens outside this file in `test_advanced_robust.py`, but uses this file for retrieval.

## Practical modification map

| Goal | First place to edit |
|---|---|
| Add new LLM backend | `RobustLLMController` and a concrete controller class |
| Change metadata extraction | `RobustMemoryNote.analyze_content` |
| Add new memory fields | `RobustMemoryNote.__init__` |
| Change memory indexing text | `RobustAgenticMemorySystem.add_note` and `consolidate_memories` |
| Change candidate neighbor retrieval | `find_related_memories` |
| Change QA-time context expansion | `find_related_memories_raw` |
| Change memory evolution strategy | `process_memory` |
| Make links stable | replace index-based links with note ids |
| Add typed/weighted graph edges | `process_memory`, `RobustMemoryNote.links`, and retrieval expansion |
| Reduce API cost | simplify `analyze_content` or reduce calls inside `process_memory` |
| Improve cache correctness | consolidate retriever after old memory updates |

## Known code issues and risks

- Some comments/docstrings contain mojibake characters due to encoding display issues.
- `json`, `re`, `Any`, and `simple_tokenize` are imported but not used in this file.
- `RobustLiteLLMController` is defined but unreachable through `RobustLLMController`.
- `api_base` is accepted by `RobustLLMController` but not used.
- `links` are stored as indices, which can become fragile if memory order changes.
- Neighbor expansion lacks deduplication and bounds checks.
- Retriever embeddings may become stale after `update_neighbor` changes old memories.
- `RobustMemoryNote.__init__` checks missing `category` before metadata extraction, but `analyze_content` does not produce category.

## Minimal conceptual summary

```text
RobustMemoryNote:
  one memory unit with content, keywords, context, tags, links

RobustAgenticMemorySystem:
  stores notes, retrieves related notes, evolves memory links and metadata

RobustLLMController:
  picks the LLM backend used by metadata extraction and evolution

SimpleEmbeddingRetriever:
  imported embedding retriever used for both memory evolution candidates and QA retrieval
```

