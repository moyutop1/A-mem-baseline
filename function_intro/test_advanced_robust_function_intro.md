# test_advanced_robust.py interface guide

This document summarizes every class method and top-level function in `test_advanced_robust.py`.
It is written as a reading and modification map for reproducing A-Mem and building extensions on top of it.

## File role

`test_advanced_robust.py` is the robust evaluation harness for A-Mem on the LoCoMo dataset.
It does not define the core memory algorithm itself. Instead, it wires together:

- LoCoMo data loading from `load_dataset.py`
- A-Mem memory construction from `memory_layer_robust.py`
- question answering with retrieved memories
- metric calculation from `utils.py`
- result logging and JSON output

High-level flow:

```text
main()
  -> evaluate_dataset()
       -> RobustAdvancedMemAgent(...)
            -> RobustAgenticMemorySystem(...)
       -> agent.add_memory(...) for each conversation turn
       -> agent.answer_question(...) for each QA item
       -> calculate_metrics(...)
       -> aggregate_metrics(...)
```

## Module-level initialization

### NLTK resource check

Purpose:

- Ensures `punkt` and `wordnet` resources are available.
- Downloads them if missing.

Important behavior:

- This runs as soon as the file is imported or executed.
- It may trigger network access if NLTK data is not already installed.

Innovation note:

- If you want this file to be import-safe in offline or cluster environments, move this setup behind `main()` or make it optional.

### `sentence_model`

Purpose:

- Tries to initialize `SentenceTransformer('all-MiniLM-L6-v2')`.

Important behavior:

- In this file, `sentence_model` is not directly used later.
- The actual retriever model is initialized inside `RobustAgenticMemorySystem`.

Innovation note:

- This looks like leftover setup. You can remove it or reuse it only after checking whether downstream code needs a shared embedding model.

## Class: `RobustAdvancedMemAgent`

Location:

```python
class RobustAdvancedMemAgent:
```

Purpose:

- A thin experimental wrapper around the robust A-Mem memory system.
- Despite the name, this is not a fully autonomous planning agent.
- It mainly provides a memory-augmented QA interface:
  - add conversation turns to memory
  - retrieve related memories
  - generate final answers from retrieved context

Main internal fields:

- `self.memory_system`: a `RobustAgenticMemorySystem`, responsible for memory storage, metadata extraction, memory evolution, and retrieval.
- `self.retriever_llm`: a `RobustLLMController`, used for query keyword generation and optional relevance filtering.
- `self.retrieve_k`: number of memories to retrieve at QA time.
- `self.temperature_c5`: decoding temperature for category 5 adversarial questions.

### `__init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000)`

Purpose:

- Initializes one memory-augmented QA agent for a single LoCoMo sample.
- Creates both the memory system and a separate LLM controller for retrieval-related prompts.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `model` | any/string-like | required | LLM model name, such as `gpt-4o-mini`, `qwen2.5:3b`, or `Qwen/Qwen2.5-3B-Instruct`. |
| `backend` | string | required | LLM backend. Expected values are compatible with `RobustLLMController`, such as `openai`, `ollama`, `sglang`, or `vllm`. |
| `retrieve_k` | int | required | Number of memories retrieved for each question. Stored in `self.retrieve_k`. |
| `temperature_c5` | float | required | Temperature used when answering category 5 adversarial questions. |
| `sglang_host` | string | `"http://localhost"` | Host for SGLang or vLLM-compatible local serving. |
| `sglang_port` | int | `30000` | Port for SGLang or vLLM-compatible local serving. |

Returns:

- No explicit return value.

Side effects:

- Instantiates `RobustAgenticMemorySystem`.
- Instantiates `RobustLLMController`.
- May load embedding models through the memory system.

Downstream calls:

- `RobustAgenticMemorySystem(...)`
- `RobustLLMController(...)`

Innovation hooks:

- Change `model_name='all-MiniLM-L6-v2'` if you want to evaluate stronger embedding models.
- Add extra constructor parameters if your innovation needs a reranker, memory compression module, temporal retriever, or graph retriever.

### `add_memory(self, content, time=None)`

Purpose:

- Adds one conversation turn into A-Mem.
- This is the write path from dataset conversation history into the memory system.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `content` | string | required | Text content to store as a memory note. In this file it is built from speaker name plus turn text. |
| `time` | string or `None` | `None` | Timestamp/date of the conversation session. Passed to the memory note as its timestamp. |

Returns:

- No explicit return value.
- Internally, `RobustAgenticMemorySystem.add_note(...)` returns a note id, but this wrapper discards it.

Side effects:

- Adds a memory note to `self.memory_system.memories`.
- Updates the retriever index.
- May call the LLM several times for metadata extraction and memory evolution.

Downstream calls:

```python
self.memory_system.add_note(content, time=time)
```

Innovation hooks:

- Return the note id if you need to debug individual memories.
- Add structured metadata here, such as speaker id, session id, evidence id, modality, or original turn id.

### `retrieve_memory(self, content, k=10)`

Purpose:

- Retrieves memory context related to a query string.
- This is the direct retrieval wrapper used during QA.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `content` | string | required | Retrieval query. In `answer_question`, this is generated keyword text rather than the original question. |
| `k` | int | `10` | Number of top memories to retrieve. |

Returns:

- A string containing retrieved memories and linked neighbor memories.

Downstream calls:

```python
self.memory_system.find_related_memories_raw(content, k=k)
```

Innovation hooks:

- Replace this with hybrid retrieval, category-aware retrieval, time-aware retrieval, graph expansion, or reranking.
- If you want better diagnostics, return both the context string and structured memory ids/scores.

### `retrieve_memory_llm(self, memories_text, query)`

Purpose:

- Uses an LLM to select the parts of retrieved memory text most relevant to a question.
- It is defined but not used in the current `answer_question` pipeline.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `memories_text` | string | required | Retrieved memory context. |
| `query` | string | required | User question. |

Returns:

- A cleaned string containing relevant memory parts, parsed by `parse_relevant_parts`.

Downstream calls:

```python
response = self.retriever_llm.llm.get_completion(prompt)
return parse_relevant_parts(response)
```

Side effects:

- Calls the LLM once.

Innovation hooks:

- You can insert this after retrieval and before final answering as an LLM-based context compressor.
- It may improve precision but increases API cost and latency.

Current status:

- Unused in the default flow.

### `generate_query_llm(self, question)`

Purpose:

- Converts a natural-language question into comma-separated retrieval keywords.
- This is the first step in the QA retrieval pipeline.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `question` | string | required | Original QA question from LoCoMo. |

Returns:

- A keyword string parsed by `parse_keywords_response`.

Downstream calls:

```python
response = self.retriever_llm.llm.get_completion(prompt)
result = parse_keywords_response(response)
```

Side effects:

- Calls the LLM once per question.

Innovation hooks:

- Replace with deterministic keyword extraction to reduce API cost.
- Use multi-query expansion for multi-hop questions.
- Use category-specific query generation, especially for temporal questions.

### `answer_question(self, question: str, category: int, answer: str) -> tuple`

Purpose:

- Runs the full QA path for one LoCoMo question.
- Generates retrieval keywords, retrieves memory context, builds a category-specific prompt, and calls the LLM for the final answer.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `question` | string | required | QA question from the dataset. |
| `category` | int | required | LoCoMo question category. Expected values: `1`, `2`, `3`, `4`, `5`. |
| `answer` | string | required | Reference answer. For category 5, this is used to construct a two-choice prompt with a distractor. |

Returns:

```python
(response, user_prompt, raw_context)
```

Return fields:

| Field | Type | Meaning |
|---|---:|---|
| `response` | string | Raw LLM answer text. It is parsed later by `parse_plain_text_answer`. |
| `user_prompt` | string | Final prompt sent to the answering LLM. Useful for debugging. |
| `raw_context` | string | Retrieved memory context used for answering. Useful for retrieval error analysis. |

Internal flow:

```text
question
  -> generate_query_llm(question)
  -> retrieve_memory(keywords, k=self.retrieve_k)
  -> build prompt according to category
  -> self.memory_system.llm_controller.llm.get_completion(...)
  -> return response, prompt, context
```

Category behavior:

| Category | Meaning from README | Prompt behavior |
|---:|---|---|
| `1` | Multi-hop | Uses generic short-phrase answer prompt. |
| `2` | Temporal | Adds instruction to use conversation dates and answer with approximate date. |
| `3` | Open-domain | Uses generic short-phrase answer prompt. |
| `4` | Single-hop | Uses generic short-phrase answer prompt. |
| `5` | Adversarial | Creates a two-choice prompt: reference answer vs `Not mentioned in the conversation`. |

Side effects:

- Calls the LLM once for keyword generation.
- Calls memory retrieval.
- Calls the LLM once for final answering.

Failure behavior:

- If final answer generation raises an exception, logs a warning and returns an empty response string.

Innovation hooks:

- Add category-aware retrieval before prompt construction.
- Replace the final prompt templates.
- Add chain-of-thought-free decomposition for category 1 multi-hop questions.
- Add temporal normalization for category 2.
- Add confidence calibration for category 5.
- Return structured retrieval diagnostics such as memory ids, scores, and link hops.

## Function: `setup_logger(log_file: Optional[str] = None) -> logging.Logger`

Purpose:

- Creates and configures a logger for evaluation.
- Logs to console and optionally to a file.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `log_file` | string or `None` | `None` | Path to the log file. If provided, a file handler is added. |

Returns:

- A configured `logging.Logger` named `locomo_eval_robust`.

Side effects:

- Adds a console handler every time it is called.
- Adds a file handler if `log_file` is provided.

Important caveat:

- Repeated calls can attach duplicate handlers to the same logger, causing duplicate log lines.

Innovation hooks:

- Add a guard to avoid duplicate handlers.
- Add debug-level logging for retrieved memory ids and scores.

## Function: `evaluate_dataset(...)`

Signature:

```python
def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000):
```

Purpose:

- Main evaluation pipeline.
- Loads LoCoMo data, builds or loads A-Mem memories for each sample, answers all QA pairs, computes metrics, writes logs, optionally saves a JSON result file, and returns the final results object.

Parameters:

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `dataset_path` | string | required | Path to the LoCoMo JSON dataset. |
| `model` | string | required | LLM model name. Passed into `RobustAdvancedMemAgent`. |
| `output_path` | string or `None` | `None` | Where to save final JSON results. If `None`, no result file is written. |
| `ratio` | float | `1.0` | Fraction of dataset samples to evaluate. Uses the first `int(len(samples) * ratio)` samples, with minimum 1 when ratio is below 1. |
| `backend` | string | `"sglang"` | LLM backend used by the agent. |
| `temperature_c5` | float | `0.5` | Temperature for category 5 adversarial QA. |
| `retrieve_k` | int | `10` | Number of memories retrieved per question. |
| `sglang_host` | string | `"http://localhost"` | Host for local SGLang/vLLM serving. |
| `sglang_port` | int | `30000` | Port for local SGLang/vLLM serving. |

Returns:

- `final_results`, a dictionary with this structure:

```python
{
    "model": model,
    "dataset": dataset_path,
    "memory_layer": "robust",
    "total_questions": total_questions,
    "category_distribution": {...},
    "aggregate_metrics": aggregate_results,
    "individual_results": results,
}
```

Major internal stages:

1. Create log file under `logs/`.
2. Load dataset via `load_locomo_dataset(dataset_path)`.
3. Apply `ratio` slicing if requested.
4. Create cache directory named `cached_memories_robust_{backend}_{model}`.
5. For each sample:
   - instantiate one `RobustAdvancedMemAgent`
   - load cached memories if available
   - otherwise add every conversation turn into A-Mem and save cache
   - iterate over all QA items
   - call `agent.answer_question(...)`
   - parse answer
   - calculate metrics
   - append per-question result
6. Aggregate metrics with `aggregate_metrics(...)`.
7. Save JSON output if `output_path` is provided.
8. Return final results.

Cache files per sample:

| File | Meaning |
|---|---|
| `memory_cache_sample_{sample_idx}.pkl` | Pickled `memory_system.memories`. |
| `retriever_cache_sample_{sample_idx}.pkl` | Pickled retriever metadata/state. |
| `retriever_cache_embeddings_sample_{sample_idx}.npy` | Saved dense embeddings for retriever. |

Important cache behavior:

- If memory cache exists, memory construction is skipped.
- If retriever cache exists, retriever is loaded directly.
- If retriever cache is missing but memory cache exists, retriever is rebuilt from local memory.
- If you change memory construction, metadata extraction, or evolution logic, delete the corresponding cache directory before rerunning.
- If you only change final answering prompts or `retrieve_k`, cache reuse is usually acceptable.

LLM call cost:

- First run can be expensive because every conversation turn may trigger metadata extraction and memory evolution calls.
- Every QA item usually triggers one query-generation LLM call and one final-answer LLM call.

Key downstream calls:

```python
samples = load_locomo_dataset(dataset_path)
agent = RobustAdvancedMemAgent(...)
agent.add_memory(...)
agent.answer_question(...)
prediction = parse_plain_text_answer(prediction)
metrics = calculate_metrics(prediction, qa.final_answer)
aggregate_results = aggregate_metrics(all_metrics, all_categories)
```

Important variables:

| Variable | Meaning |
|---|---|
| `samples` | Loaded LoCoMo samples. |
| `agent` | One memory-augmented QA wrapper per sample. |
| `memories_dir` | Cache directory for this backend/model pair. |
| `raw_context` | Retrieved memory context used to answer a question. |
| `user_prompt` | Final answering prompt sent to the LLM. |
| `metrics` | Per-question evaluation metrics. |
| `aggregate_results` | Overall and category-level metric summaries. |

Known caveats:

- `i = 0` is unused.
- `error_num` is logged but never incremented.
- If no questions are evaluated, the category summary can divide by zero.
- `output_path` is written without explicit UTF-8 encoding.
- Cache directory names include raw model strings; model names containing `/` can create nested directories.

Innovation hooks:

- Add new result fields to `individual_results`, such as retrieved ids, scores, hop depth, or context length.
- Add ablation switches, such as disabling memory evolution or disabling neighbor expansion.
- Add category-specific retrieval strategies.
- Add richer error analysis output for failed QA cases.

## Function: `main()`

Purpose:

- Command-line entry point.
- Parses CLI arguments, validates `ratio`, resolves paths relative to this file, and calls `evaluate_dataset`.

Parameters:

- No direct Python parameters.
- Reads arguments from the command line through `argparse`.

CLI arguments:

| Argument | Type | Default | Meaning |
|---|---:|---:|---|
| `--dataset` | string | `data/locomo10.json` | Dataset path relative to this file unless an absolute path is passed. |
| `--model` | string | `gpt-4o-mini` | LLM model name. |
| `--output` | string or `None` | `None` | Output JSON path relative to this file. |
| `--ratio` | float | `1.0` | Fraction of dataset to evaluate. Must be in `(0.0, 1.0]`. |
| `--backend` | string | `openai` | Backend to use: `openai`, `ollama`, `sglang`, or `vllm`. |
| `--temperature_c5` | float | `0.5` | Temperature for category 5 questions. |
| `--retrieve_k` | int | `10` | Number of memories to retrieve. |
| `--sglang_host` | string | `http://localhost` | Local serving host. |
| `--sglang_port` | int | `30000` | Local serving port. |

Returns:

- No explicit return value.

Raises:

- `ValueError` if `ratio <= 0.0` or `ratio > 1.0`.

Example:

```bash
python test_advanced_robust.py \
  --backend openai \
  --model gpt-4o-mini \
  --dataset data/locomo10.json \
  --ratio 0.1 \
  --retrieve_k 10 \
  --output quick.json
```

Innovation hooks:

- Add CLI flags for ablations, new retrievers, prompt variants, cache reset, or debug output.
- Add a `--max_questions` flag for faster development.
- Add a `--no_cache` flag to force memory reconstruction.

## Practical modification map

Use this map when deciding where to modify the code:

| Goal | First place to edit |
|---|---|
| Change how turns are written into memory | `evaluate_dataset`, where `conversation_tmp` is built |
| Add speaker/session/evidence metadata | `add_memory` and the call site in `evaluate_dataset` |
| Reduce API cost | `generate_query_llm`, `retrieve_memory_llm`, memory construction cache policy |
| Change retrieval depth | CLI `--retrieve_k`, `retrieve_memory`, or memory system retriever |
| Add reranking or context compression | `retrieve_memory` or activate `retrieve_memory_llm` |
| Change final answer prompt | `answer_question` |
| Add category-specific strategies | `answer_question` and `evaluate_dataset` QA loop |
| Add new metrics | `evaluate_dataset` around `calculate_metrics` |
| Add richer JSON output | `result` dict inside `evaluate_dataset` |
| Add ablation experiments | `main` CLI args and `evaluate_dataset` control flow |

## API cost summary

The expensive parts are:

```text
memory construction:
  number of conversation turns * metadata/evolution LLM calls

QA:
  number of questions * (query generation call + final answer call)
```

Recommended development command:

```bash
python test_advanced_robust.py \
  --backend openai \
  --model gpt-4o-mini \
  --dataset data/locomo10.json \
  --ratio 0.1 \
  --retrieve_k 5 \
  --output quick.json
```

Before evaluating a memory-system change, delete the matching cache directory:

```text
cached_memories_robust_{backend}_{model}
```
