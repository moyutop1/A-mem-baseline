"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any
import json
import re
import uuid
import os
import time
import logging
import functools
from datetime import datetime
from abc import ABC, abstractmethod

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
    MEMORY_RELATION_RESOLUTION_PROMPT,
    MEMORY_REWRITE_PROMPT,
    DOMAIN_RERANK_PROMPT,
    parse_memory_relation_resolution,
    parse_memory_rewrite,
    parse_domain_rerank,
)

logger = logging.getLogger("amem_robust")

STABLE_RETRIEVAL_EDGES = {
    "semantic_related",
    "same_topic",
    "same_entity",
    "evidence_for",
    "generalizes",
    "elaborates",
    "co_used",
}
DISALLOWED_GRAPH_EDGES = {"updates", "replaces", "conflicts_with", "update", "replace", "conflict"}
ACTIVE_RETRIEVAL_STATUSES = {"active", "candidate", "stale"}
DEFAULT_MEMORY_LEVEL = "instance"
DEFAULT_DOMAIN_CANDIDATE_TOP_K = 3
DEFAULT_DOMAIN_EMBEDDING_THRESHOLD = 0.25

UNCERTAIN_MARKERS = {
    "maybe", "might", "possibly", "probably", "not sure", "uncertain",
    "temporary", "temporarily", "one-time", "one time", "for this task",
}
SUPERSESSION_MARKERS = {
    "now", "currently", "decided", "instead", "rather than", "changed",
    "switch", "switched", "no longer", "replaced", "supersedes",
}

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        #self.client = OpenAI(api_key=api_key)
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        if response is None or not getattr(response, "choices", None):
            raise RuntimeError("OpenAI-compatible backend returned an empty response")
        return response.choices[0].message.content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 domain_candidate_top_k: int = DEFAULT_DOMAIN_CANDIDATE_TOP_K,
                 domain_embedding_threshold: float = DEFAULT_DOMAIN_EMBEDDING_THRESHOLD):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 current_content: Optional[str] = None,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[List[Any]] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 domain_paths: Optional[List[str]] = None,
                 memory_level: Optional[str] = None,
                 status: Optional[str] = None,
                 version: Optional[int] = None,
                 conditions: Optional[List[Dict[str, Any]]] = None,
                 revision_history: Optional[List[Dict[str, Any]]] = None,
                 evidence_memory_ids: Optional[List[str]] = None,
                 confidence: Optional[float] = None,
                 last_updated: Optional[str] = None,
                 llm_controller: Optional[RobustLLMController] = None):

        self.content = current_content or content
        self.current_content = self.content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.last_updated = last_updated or self.last_accessed

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.domain_paths = domain_paths or self._infer_domain_paths()
        self.memory_level = (memory_level or DEFAULT_MEMORY_LEVEL).lower()
        if self.memory_level not in {"instance", "task", "generalized"}:
            self.memory_level = DEFAULT_MEMORY_LEVEL
        self.status = (status or "active").lower()
        if self.status not in {"active", "candidate", "stale", "deprecated", "archived"}:
            self.status = "active"
        self.version = int(version or 1)
        self.conditions = conditions or []
        self.revision_history = revision_history or []
        self.evidence_memory_ids = evidence_memory_ids or []
        self.confidence = float(confidence if confidence is not None else 1.0)

    def _infer_domain_paths(self) -> List[str]:
        """Create lightweight domain paths from existing category/tags metadata."""
        paths = []
        if self.category and self.category != "Uncategorized":
            paths.append(str(self.category))
        for tag in self.tags[:3]:
            if tag and tag not in paths:
                paths.append(str(tag))
        return paths or ["General"]

    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False):

        self.memories: Dict[str, RobustMemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.domain_candidate_top_k = max(1, int(domain_candidate_top_k))
        self.domain_embedding_threshold = float(domain_embedding_threshold)

    def _memory_to_index_text(self, memory: RobustMemoryNote) -> str:
        """Build the retriever document from the current valid memory object."""
        return (
            "content:" + getattr(memory, "current_content", memory.content) +
            " context:" + getattr(memory, "context", "General") +
            " keywords: " + ", ".join(getattr(memory, "keywords", [])) +
            " tags: " + ", ".join(getattr(memory, "tags", [])) +
            " domains: " + ", ".join(getattr(memory, "domain_paths", [])) +
            " status: " + getattr(memory, "status", "active")
        )

    def _domain_query_text(self, text: str, note: Optional[RobustMemoryNote] = None) -> str:
        if note is None:
            return text
        return " ".join([
            text,
            getattr(note, "context", ""),
            " ".join(getattr(note, "keywords", [])),
            " ".join(getattr(note, "tags", [])),
        ]).strip()

    def _domain_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Build a lightweight memory domain tree catalog from stored domain paths."""
        catalog: Dict[str, Dict[str, Any]] = {}
        for memory in self.memories.values():
            memory = self._ensure_memory_schema(memory)
            for domain_path in getattr(memory, "domain_paths", []) or ["General"]:
                parts = [part.strip() for part in str(domain_path).split("/") if part.strip()]
                if not parts:
                    parts = ["General"]
                for depth in range(1, len(parts) + 1):
                    path = " / ".join(parts[:depth])
                    entry = catalog.setdefault(path, {"count": 0, "examples": []})
                    entry["count"] += 1
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(getattr(memory, "current_content", memory.content))
        return catalog

    def _cosine_similarity(self, left: Any, right: Any) -> float:
        left_vec = list(left)
        right_vec = list(right)
        if not left_vec or not right_vec:
            return 0.0
        numerator = sum(float(a) * float(b) for a, b in zip(left_vec, right_vec))
        left_norm = sum(float(a) * float(a) for a in left_vec) ** 0.5
        right_norm = sum(float(b) * float(b) for b in right_vec) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _rank_domain_candidates_by_embedding(
        self,
        text: str,
        max_candidates: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Embedding-rank domain candidates, thresholded but always returning at least one."""
        catalog = self._domain_catalog()
        if not catalog:
            return []

        max_candidates = max_candidates or self.domain_candidate_top_k
        threshold = self.domain_embedding_threshold if threshold is None else threshold
        domain_paths = list(catalog.keys())
        domain_texts = []
        for path in domain_paths:
            examples = " ".join(catalog[path]["examples"])
            domain_texts.append(f"domain path: {path} examples: {examples}")

        try:
            query_embedding = self.retriever.model.encode([text])[0]
            domain_embeddings = self.retriever.model.encode(domain_texts)
            scored = []
            for path, domain_text, embedding in zip(domain_paths, domain_texts, domain_embeddings):
                scored.append({
                    "domain_path": path,
                    "score": self._cosine_similarity(query_embedding, embedding),
                    "text": domain_text,
                    "count": catalog[path]["count"],
                })
        except Exception as e:
            logger.warning("Domain embedding ranking failed: %s; using lexical fallback", e)
            query_tokens = set(re.findall(r"\b[\w\-]{3,}\b", text.lower()))
            scored = []
            for path, domain_text in zip(domain_paths, domain_texts):
                domain_tokens = set(re.findall(r"\b[\w\-]{3,}\b", domain_text.lower()))
                score = len(query_tokens & domain_tokens) / max(1, len(query_tokens | domain_tokens))
                scored.append({
                    "domain_path": path,
                    "score": score,
                    "text": domain_text,
                    "count": catalog[path]["count"],
                })

        scored.sort(key=lambda item: (item["score"], item["count"]), reverse=True)
        selected = [item for item in scored if item["score"] >= threshold][:max_candidates]
        if not selected and scored:
            selected = [scored[0]]
        return selected[:max_candidates]

    def route_domains(
        self,
        text: str,
        note: Optional[RobustMemoryNote] = None,
        max_domains: Optional[int] = None,
    ) -> List[str]:
        """Route text to memory domains: embedding candidates followed by LLM rerank."""
        max_domains = max_domains or self.domain_candidate_top_k
        query_text = self._domain_query_text(text, note)
        candidates = self._rank_domain_candidates_by_embedding(query_text, max_candidates=max_domains)
        if not candidates:
            return getattr(note, "domain_paths", None) or ["General"]

        candidate_paths = [candidate["domain_path"] for candidate in candidates]
        candidate_text = "\n".join(
            f"{idx}. {candidate['domain_path']} (embedding_score={candidate['score']:.4f})"
            for idx, candidate in enumerate(candidates)
        )
        try:
            prompt = DOMAIN_RERANK_PROMPT.format(
                text=query_text,
                candidate_domains=candidate_text,
                max_domains=max_domains,
            )
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            reranked = parse_domain_rerank(response, candidate_paths, max_domains=max_domains)
            domains = reranked["domains"]
        except Exception as e:
            logger.warning("Domain LLM rerank failed: %s; using embedding domain order", e)
            domains = candidate_paths
        return domains[:max_domains] or [candidate_paths[0]]

    def _domain_overlap(self, memory: RobustMemoryNote, routed_domains: Optional[List[str]]) -> bool:
        if not routed_domains:
            return True
        memory_domains = getattr(memory, "domain_paths", []) or ["General"]
        for routed in routed_domains:
            for domain in memory_domains:
                if routed == domain or routed.startswith(domain + " / ") or domain.startswith(routed + " / "):
                    return True
        return False

    def _is_indexable(self, memory: RobustMemoryNote) -> bool:
        return getattr(memory, "status", "active") in ACTIVE_RETRIEVAL_STATUSES

    def _is_retrievable(self, memory: RobustMemoryNote, query: str = "") -> bool:
        status = getattr(memory, "status", "active")
        if status in {"deprecated", "archived"}:
            historical_markers = ("previous", "formerly", "old", "history", "historical", "archived")
            return any(marker in query.lower() for marker in historical_markers)
        return status in ACTIVE_RETRIEVAL_STATUSES

    def lifecycle_penalty(self, memory: RobustMemoryNote) -> float:
        """Penalty used by lifecycle-aware retrieval/reranking."""
        status = getattr(memory, "status", "active")
        if status == "active":
            return 0.0
        if status == "candidate":
            return 0.2
        if status == "stale":
            return 0.4
        if status == "deprecated":
            return 0.9
        if status == "archived":
            return 0.8
        return 0.0

    def _ensure_memory_schema(self, memory: RobustMemoryNote) -> RobustMemoryNote:
        """Backfill lifecycle fields for old pickled caches or legacy notes."""
        if not hasattr(memory, "current_content"):
            memory.current_content = memory.content
        if not hasattr(memory, "domain_paths"):
            memory.domain_paths = ["General"]
        if not hasattr(memory, "memory_level"):
            memory.memory_level = DEFAULT_MEMORY_LEVEL
        if not hasattr(memory, "status"):
            memory.status = "active"
        if not hasattr(memory, "version"):
            memory.version = 1
        if not hasattr(memory, "conditions"):
            memory.conditions = []
        if not hasattr(memory, "revision_history"):
            memory.revision_history = []
        if not hasattr(memory, "evidence_memory_ids"):
            memory.evidence_memory_ids = []
        if not hasattr(memory, "confidence"):
            memory.confidence = 1.0
        if not hasattr(memory, "last_updated"):
            memory.last_updated = getattr(memory, "last_accessed", getattr(memory, "timestamp", ""))
        return memory

    def _format_memory_for_context(self, memory: RobustMemoryNote, relation: Optional[str] = None) -> str:
        relation_text = f"relation: {relation} " if relation else ""
        return (
            relation_text +
            "talk start time:" + str(memory.timestamp) +
            " memory content: " + str(memory.current_content) +
            " memory context: " + str(memory.context) +
            " memory keywords: " + str(memory.keywords) +
            " memory tags: " + str(memory.tags) +
            " memory status: " + str(memory.status) +
            " memory confidence: " + str(memory.confidence) + "\n"
        )

    def _edge_relation(self, edge: Any) -> str:
        if isinstance(edge, dict):
            relation = str(edge.get("relation", "semantic_related")).lower()
        else:
            relation = "semantic_related"
        if relation in DISALLOWED_GRAPH_EDGES:
            return "none"
        if relation not in STABLE_RETRIEVAL_EDGES:
            return "semantic_related"
        return relation

    def _edge_to_memory(
        self,
        edge: Any,
        all_memories: List[RobustMemoryNote],
        id_to_memory: Dict[str, RobustMemoryNote],
    ) -> Optional[RobustMemoryNote]:
        if isinstance(edge, dict):
            target_id = edge.get("target_id")
            if target_id:
                return id_to_memory.get(target_id)
            target_index = edge.get("target_index")
            if isinstance(target_index, int) and 0 <= target_index < len(all_memories):
                return self._ensure_memory_schema(all_memories[target_index])
            return None
        if isinstance(edge, int) and 0 <= edge < len(all_memories):
            return self._ensure_memory_schema(all_memories[edge])
        return None

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note using write-time lifecycle consolidation."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        note.domain_paths = self.route_domains(note.current_content, note=note)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        if getattr(note, "_requires_reindex", False):
            self.consolidate_memories()
        else:
            self.retriever.add_documents([self._memory_to_index_text(note)])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            if self._is_indexable(memory):
                self.retriever.add_documents([self._memory_to_index_text(memory)])

    def find_related_memories(
        self,
        query: str,
        k: int = 5,
        routed_domains: Optional[List[str]] = None,
        route_domain: bool = True,
    ) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        if route_domain and routed_domains is None:
            routed_domains = self.route_domains(query)
        indices = self.retriever.search(query, min(k * 3, max(k, len(self.memories))))
        all_memories = list(self.memories.values())
        memory_str = ""
        ranked = []
        for order, i in enumerate(indices):
            if i >= len(all_memories):
                continue
            memory = self._ensure_memory_schema(all_memories[i])
            if not self._is_retrievable(memory, query):
                continue
            if not self._domain_overlap(memory, routed_domains):
                continue
            ranked.append((self.lifecycle_penalty(memory), order, i, memory))

        if not ranked and routed_domains:
            for order, i in enumerate(indices):
                if i >= len(all_memories):
                    continue
                memory = self._ensure_memory_schema(all_memories[i])
                if not self._is_retrievable(memory, query):
                    continue
                ranked.append((self.lifecycle_penalty(memory) + 0.1, order, i, memory))

        ranked.sort(key=lambda item: (item[0], item[1]))
        filtered_indices = []
        for _, _, i, memory in ranked[:k]:
            filtered_indices.append(i)
            memory_str += (
                "memory index:" + str(i) +
                "\t memory id:" + memory.id +
                "\t talk start time:" + memory.timestamp +
                "\t memory content: " + memory.current_content +
                "\t memory context: " + memory.context +
                "\t memory keywords: " + str(memory.keywords) +
                "\t memory tags: " + str(memory.tags) +
                "\t memory status: " + memory.status +
                "\t memory confidence: " + str(memory.confidence) + "\n"
            )
        return memory_str, filtered_indices

    def find_related_memories_raw(
        self,
        query: str,
        k: int = 5,
        routed_domains: Optional[List[str]] = None,
        route_domain: bool = True,
    ) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        if route_domain and routed_domains is None:
            routed_domains = self.route_domains(query)
        indices = self.retriever.search(query, min(k * 3, max(k, len(self.memories))))
        all_memories = list(self.memories.values())
        id_to_memory = {memory.id: self._ensure_memory_schema(memory) for memory in all_memories}
        memory_str = ""
        seen_ids = set()
        ranked = []
        for order, i in enumerate(indices):
            if i >= len(all_memories):
                continue
            memory = self._ensure_memory_schema(all_memories[i])
            if not self._is_retrievable(memory, query):
                continue
            if not self._domain_overlap(memory, routed_domains):
                continue
            ranked.append((self.lifecycle_penalty(memory), order, i, memory))

        if not ranked and routed_domains:
            for order, i in enumerate(indices):
                if i >= len(all_memories):
                    continue
                memory = self._ensure_memory_schema(all_memories[i])
                if not self._is_retrievable(memory, query):
                    continue
                ranked.append((self.lifecycle_penalty(memory) + 0.1, order, i, memory))

        ranked.sort(key=lambda item: (item[0], item[1]))
        for _, _, i, memory in ranked:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            memory_str += self._format_memory_for_context(memory)

            neighbor_count = 0
            for edge in getattr(memory, "links", []):
                target_memory = self._edge_to_memory(edge, all_memories, id_to_memory)
                relation = self._edge_relation(edge)
                if (
                    not target_memory
                    or relation not in STABLE_RETRIEVAL_EDGES
                    or not self._is_retrievable(target_memory, query)
                    or not self._domain_overlap(target_memory, routed_domains)
                    or target_memory.id in seen_ids
                ):
                    continue
                seen_ids.add(target_memory.id)
                memory_str += self._format_memory_for_context(target_memory, relation=relation)
                neighbor_count += 1
                if neighbor_count >= k:
                    break
            if len(seen_ids) >= k:
                break
        return memory_str

    # ---- write-time lifecycle consolidation ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Resolve updates/conflicts at write time and maintain only stable graph edges."""
        candidate_memory, indices = self.retrieve_same_entity_or_topic(note, top_k=10)
        if not indices:
            note._requires_reindex = False
            return False, note

        try:
            decision = self.resolve_memory_relation(note, candidate_memory, indices)
            logger.debug("Write-time memory relation decision: %s", decision)
            action = decision["action"]

            if action == "mark_candidate_uncertain":
                note.status = "candidate"
                note.confidence = min(note.confidence, 0.4)
                self.add_stable_graph_edges(note, indices, decision)
                note._requires_reindex = False
                return True, note

            if action == "create_separate_memory":
                self.add_stable_graph_edges(note, indices, decision)
                note._requires_reindex = False
                return False, note

            target_index = decision.get("target")
            all_memories = list(self.memories.values())
            if target_index is None or target_index >= len(all_memories):
                self.add_stable_graph_edges(note, indices, decision)
                note._requires_reindex = False
                return False, note

            target_memory = self._ensure_memory_schema(all_memories[target_index])
            rewritten = self.rewrite_memory(target_memory, note, action, decision)
            remaining_indices = [idx for idx in indices if idx != target_index]
            self.add_stable_graph_edges(rewritten, remaining_indices, decision)
            rewritten._requires_reindex = True
            return True, rewritten

        except Exception as e:
            logger.error("Write-time consolidation failed for note %s: %s; storing separately", note.id, e)
            note._requires_reindex = False
            return False, note

    def retrieve_same_entity_or_topic(self, new_memory: RobustMemoryNote, top_k: int = 10) -> tuple:
        """Retrieve candidate memories likely to share an entity, topic, or semantic scope."""
        query = " ".join([
            new_memory.current_content,
            new_memory.context,
            " ".join(new_memory.keywords),
            " ".join(new_memory.tags),
        ])
        routed_domains = self.route_domains(query, note=new_memory)
        new_memory.domain_paths = routed_domains
        return self.find_related_memories(
            query,
            k=top_k,
            routed_domains=routed_domains,
            route_domain=False,
        )

    def resolve_memory_relation(
        self,
        new_memory: RobustMemoryNote,
        candidate_memories: str,
        candidate_indices: List[int],
    ) -> Dict[str, Any]:
        """Choose the write-time lifecycle action for a new memory candidate."""
        if not candidate_indices:
            return self._normalize_decision({"action": "create_separate_memory"}, candidate_indices)

        try:
            prompt = MEMORY_RELATION_RESOLUTION_PROMPT.format(
                content=new_memory.current_content,
                context=new_memory.context,
                keywords=new_memory.keywords,
                tags=new_memory.tags,
                candidate_memories=candidate_memories,
            )
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.2)
            decision = parse_memory_relation_resolution(response)
        except Exception as e:
            logger.warning("Relation resolution LLM failed: %s; using heuristic decision", e)
            decision = self._heuristic_relation_decision(new_memory, candidate_indices)
        return self._normalize_decision(decision, candidate_indices)

    def _normalize_decision(self, decision: Dict[str, Any], candidate_indices: List[int]) -> Dict[str, Any]:
        action = decision.get("action", "create_separate_memory")
        if action not in {
            "overwrite_existing",
            "refine_conditions",
            "append_detail",
            "create_separate_memory",
            "mark_candidate_uncertain",
        }:
            action = "create_separate_memory"

        target = decision.get("target")
        if action in {"overwrite_existing", "refine_conditions", "append_detail"}:
            if target not in candidate_indices:
                target = candidate_indices[0] if candidate_indices else None
            if target is None:
                action = "create_separate_memory"

        relation = decision.get("relation", "semantic_related")
        if relation in DISALLOWED_GRAPH_EDGES:
            relation = "semantic_related"
        if relation not in STABLE_RETRIEVAL_EDGES and relation != "none":
            relation = "semantic_related"

        try:
            confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6

        return {
            "action": action,
            "target": target,
            "relation": relation,
            "confidence": confidence,
            "reason": decision.get("reason", ""),
        }

    def _heuristic_relation_decision(
        self,
        new_memory: RobustMemoryNote,
        candidate_indices: List[int],
    ) -> Dict[str, Any]:
        """Deterministic fallback when relation-resolution LLM output is unavailable."""
        best_index, best_overlap = self._best_candidate_overlap(new_memory, candidate_indices)
        content_lower = new_memory.current_content.lower()

        if any(marker in content_lower for marker in UNCERTAIN_MARKERS):
            return {
                "action": "mark_candidate_uncertain",
                "target": best_index,
                "relation": "semantic_related",
                "confidence": 0.4,
                "reason": "The new memory contains uncertainty or temporary-scope language.",
            }

        if best_index is None or best_overlap < 0.08:
            return {
                "action": "create_separate_memory",
                "target": None,
                "relation": "semantic_related",
                "confidence": 0.6,
                "reason": "No sufficiently similar existing memory was found.",
            }

        if any(marker in content_lower for marker in SUPERSESSION_MARKERS) and best_overlap >= 0.16:
            return {
                "action": "overwrite_existing",
                "target": best_index,
                "relation": "same_entity",
                "confidence": 0.75,
                "reason": "The new memory appears to supersede a related existing memory.",
            }

        if self._looks_like_condition_refinement(new_memory, best_index):
            return {
                "action": "refine_conditions",
                "target": best_index,
                "relation": "same_topic",
                "confidence": 0.7,
                "reason": "Related preferences or facts appear to apply under different conditions.",
            }

        if best_overlap >= 0.16:
            return {
                "action": "append_detail",
                "target": best_index,
                "relation": "elaborates",
                "confidence": 0.65,
                "reason": "The new memory adds related detail to an existing memory.",
            }

        return {
            "action": "create_separate_memory",
            "target": None,
            "relation": "semantic_related",
            "confidence": 0.6,
            "reason": "The relation is weak enough to keep the memory separate.",
        }

    def _best_candidate_overlap(self, new_memory: RobustMemoryNote, candidate_indices: List[int]) -> tuple:
        all_memories = list(self.memories.values())
        new_tokens = self._memory_token_set(new_memory)
        best_index = None
        best_overlap = 0.0
        for idx in candidate_indices:
            if idx >= len(all_memories):
                continue
            old_tokens = self._memory_token_set(self._ensure_memory_schema(all_memories[idx]))
            if not old_tokens:
                continue
            overlap = len(new_tokens & old_tokens) / max(1, len(new_tokens | old_tokens))
            if overlap > best_overlap:
                best_index = idx
                best_overlap = overlap
        return best_index, best_overlap

    def _memory_token_set(self, memory: RobustMemoryNote) -> set:
        text = " ".join([
            getattr(memory, "current_content", memory.content),
            getattr(memory, "context", ""),
            " ".join(getattr(memory, "keywords", [])),
            " ".join(getattr(memory, "tags", [])),
        ])
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "into", "speaker",
            "says", "said", "user", "about", "would", "could", "should", "have",
        }
        return {
            token.lower()
            for token in re.findall(r"\b[\w\-]{3,}\b", text)
            if token.lower() not in stop_words
        }

    def _looks_like_condition_refinement(self, new_memory: RobustMemoryNote, target_index: int) -> bool:
        all_memories = list(self.memories.values())
        if target_index >= len(all_memories):
            return False
        old_memory = self._ensure_memory_schema(all_memories[target_index])
        combined = f"{old_memory.current_content} {new_memory.current_content}".lower()
        condition_terms = {
            "prefer", "prefers", "preference", "likes", "wants", "interview",
            "essay", "academic", "when", "while", "context", "under",
        }
        return sum(1 for term in condition_terms if term in combined) >= 2

    def rewrite_memory(
        self,
        old_memory: RobustMemoryNote,
        new_memory: RobustMemoryNote,
        mode: str,
        decision: Dict[str, Any],
    ) -> RobustMemoryNote:
        """Rewrite an existing memory object instead of adding update/conflict edges."""
        old_memory = self._ensure_memory_schema(old_memory)
        previous_content = old_memory.current_content
        previous_version = old_memory.version
        reason = decision.get("reason") or f"Write-time consolidation via {mode}."

        try:
            prompt = MEMORY_REWRITE_PROMPT.format(
                mode=mode,
                reason=reason,
                old_content=old_memory.current_content,
                old_context=old_memory.context,
                old_conditions=old_memory.conditions,
                new_content=new_memory.current_content,
                new_context=new_memory.context,
            )
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.2)
            rewritten = parse_memory_rewrite(response)
        except Exception as e:
            logger.warning("Memory rewrite LLM failed: %s; using deterministic rewrite", e)
            rewritten = self._deterministic_rewrite(old_memory, new_memory, mode, reason)

        content = rewritten.get("content") or self._deterministic_rewrite(
            old_memory, new_memory, mode, reason
        )["content"]
        old_memory.content = content
        old_memory.current_content = content
        old_memory.context = new_memory.context or old_memory.context
        old_memory.keywords = self._merge_unique(old_memory.keywords, new_memory.keywords)
        old_memory.tags = self._merge_unique(old_memory.tags, new_memory.tags)
        old_memory.domain_paths = self._merge_unique(
            old_memory.domain_paths,
            rewritten.get("domain_paths", []) or new_memory.domain_paths,
        )
        old_memory.memory_level = rewritten.get("memory_level") or old_memory.memory_level
        old_memory.conditions = rewritten.get("conditions") or self._merged_conditions(old_memory, new_memory, mode)
        old_memory.confidence = max(
            min(float(rewritten.get("confidence", old_memory.confidence)), 1.0),
            min(old_memory.confidence, new_memory.confidence),
        )
        old_memory.status = "active"
        old_memory.version = previous_version + 1
        old_memory.last_updated = datetime.now().strftime("%Y%m%d%H%M")
        old_memory.evidence_memory_ids = self._merge_unique(old_memory.evidence_memory_ids, [new_memory.id])
        self.append_revision_history(
            old_memory,
            previous_version=previous_version,
            previous_content=previous_content,
            new_content=content,
            update_reason=rewritten.get("update_reason") or reason,
            source_memory_id=new_memory.id,
        )
        return old_memory

    def _deterministic_rewrite(
        self,
        old_memory: RobustMemoryNote,
        new_memory: RobustMemoryNote,
        mode: str,
        reason: str,
    ) -> Dict[str, Any]:
        if mode == "overwrite_existing":
            content = new_memory.current_content
        elif mode == "refine_conditions":
            content = (
                f"{old_memory.current_content} Also, under a different condition or context, "
                f"{new_memory.current_content}"
            )
        else:
            if new_memory.current_content in old_memory.current_content:
                content = old_memory.current_content
            else:
                content = f"{old_memory.current_content} {new_memory.current_content}"
        return {
            "content": content,
            "conditions": self._merged_conditions(old_memory, new_memory, mode),
            "domain_paths": self._merge_unique(old_memory.domain_paths, new_memory.domain_paths),
            "memory_level": old_memory.memory_level,
            "confidence": min(1.0, max(old_memory.confidence, new_memory.confidence)),
            "update_reason": reason,
        }

    def _merged_conditions(
        self,
        old_memory: RobustMemoryNote,
        new_memory: RobustMemoryNote,
        mode: str,
    ) -> List[Dict[str, Any]]:
        conditions = list(old_memory.conditions)
        if mode == "overwrite_existing":
            conditions = []
        conditions.extend(new_memory.conditions)
        if not new_memory.conditions and mode in {"refine_conditions", "overwrite_existing"}:
            conditions.append({
                "context": new_memory.context,
                "detail": new_memory.current_content,
            })
        return conditions

    def append_revision_history(
        self,
        memory: RobustMemoryNote,
        previous_version: int,
        previous_content: str,
        new_content: str,
        update_reason: str,
        source_memory_id: str,
    ) -> None:
        """Append old and new versions to the memory object's revision history."""
        if not memory.revision_history:
            memory.revision_history.append({
                "version": previous_version,
                "content": previous_content,
                "update_reason": "Initial memory before write-time consolidation.",
                "timestamp": getattr(memory, "timestamp", ""),
            })
        elif memory.revision_history[-1].get("content") != previous_content:
            memory.revision_history.append({
                "version": previous_version,
                "content": previous_content,
                "update_reason": "Snapshot before write-time consolidation.",
                "timestamp": memory.last_updated,
            })
        memory.revision_history.append({
            "version": memory.version,
            "content": new_content,
            "update_reason": update_reason,
            "source_memory_id": source_memory_id,
            "timestamp": memory.last_updated,
        })

    def add_stable_graph_edges(
        self,
        source_memory: RobustMemoryNote,
        candidate_indices: List[int],
        decision: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add only stable retrieval edges; never update/replacement/conflict edges."""
        relation = (decision or {}).get("relation", "semantic_related")
        if relation == "none" or relation in DISALLOWED_GRAPH_EDGES:
            relation = "semantic_related"
        if relation not in STABLE_RETRIEVAL_EDGES:
            relation = "semantic_related"
        reason = (decision or {}).get("reason", "")
        strength = (decision or {}).get("confidence", 0.6)
        all_memories = list(self.memories.values())

        for idx in candidate_indices[:5]:
            if idx >= len(all_memories):
                continue
            target = self._ensure_memory_schema(all_memories[idx])
            if target.id == source_memory.id or not self._is_indexable(target):
                continue
            edge_relation = relation if relation in STABLE_RETRIEVAL_EDGES else self._infer_stable_relation(source_memory, target)
            self._add_edge(source_memory, target, edge_relation, reason, strength)
            if edge_relation in {"semantic_related", "same_topic", "same_entity", "co_used"}:
                self._add_edge(target, source_memory, edge_relation, reason, strength)

    def _add_edge(
        self,
        source: RobustMemoryNote,
        target: RobustMemoryNote,
        relation: str,
        reason: str,
        strength: float,
    ) -> None:
        if relation in DISALLOWED_GRAPH_EDGES or relation not in STABLE_RETRIEVAL_EDGES:
            return
        source.links = [
            edge for edge in getattr(source, "links", [])
            if not (isinstance(edge, dict) and edge.get("relation") in DISALLOWED_GRAPH_EDGES)
        ]
        for edge in source.links:
            if isinstance(edge, dict) and edge.get("target_id") == target.id and edge.get("relation") == relation:
                edge["strength"] = max(float(edge.get("strength", 0.0)), float(strength))
                return
        source.links.append({
            "target_id": target.id,
            "relation": relation,
            "strength": float(strength),
            "reason": reason,
            "created_at": datetime.now().strftime("%Y%m%d%H%M"),
        })

    def _infer_stable_relation(self, source: RobustMemoryNote, target: RobustMemoryNote) -> str:
        if set(source.tags) & set(target.tags):
            return "same_topic"
        if set(source.keywords) & set(target.keywords):
            return "semantic_related"
        return "semantic_related"

    def _merge_unique(self, left: List[Any], right: List[Any]) -> List[Any]:
        merged = []
        for item in list(left or []) + list(right or []):
            if item and item not in merged:
                merged.append(item)
        return merged

    # ---- legacy evolution implementation kept for reference ----

    def _legacy_process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note
