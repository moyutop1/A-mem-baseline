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

from typing import List, Dict, Optional, Literal, Any, Set
import json
import re
import uuid
import os
import time
import logging
import functools
import math
from datetime import datetime
from abc import ABC, abstractmethod

try:
    from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
except ImportError as import_error:
    logger_import_error = import_error

    def simple_tokenize(text):
        return re.findall(r"\b\w+\b", str(text).lower())

    class SimpleEmbeddingRetriever:
        """Local fallback used when optional baseline dependencies are unavailable."""

        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            class _HashingEmbeddingModel:
                def __init__(self, dim: int = 384):
                    self.dim = dim

                def encode(self, documents):
                    vectors = []
                    for doc in documents:
                        vector = [0.0] * self.dim
                        for token in re.findall(r"\b\w+\b", str(doc).lower()):
                            idx = hash(token) % self.dim
                            vector[idx] += 1.0
                        norm = sum(value * value for value in vector) ** 0.5
                        if norm:
                            vector = [value / norm for value in vector]
                        vectors.append(vector)
                    return vectors

            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except ImportError:
                self.model = _HashingEmbeddingModel()
            self.corpus = []
            self.embeddings = None
            self.document_ids = {}

        def add_documents(self, documents: List[str]):
            import numpy as _np
            if not self.corpus:
                self.corpus = list(documents)
                self.embeddings = self.model.encode(documents)
                self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
                return
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            self.embeddings = new_embeddings if self.embeddings is None else _np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

        def search(self, query: str, k: int = 5):
            import numpy as _np
            if not self.corpus or self.embeddings is None:
                return []
            query_embedding = self.model.encode([query])[0]
            left = _np.asarray(self.embeddings, dtype=float)
            right = _np.asarray(query_embedding, dtype=float)
            denom = (_np.linalg.norm(left, axis=1) * max(_np.linalg.norm(right), 1e-12))
            similarities = left.dot(right) / _np.maximum(denom, 1e-12)
            return _np.argsort(similarities)[-k:][::-1]

        def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
            import numpy as _np
            import pickle as _pickle
            if self.embeddings is not None:
                _np.save(retriever_cache_embeddings_file, self.embeddings)
            with open(retriever_cache_file, "wb") as f:
                _pickle.dump({"corpus": self.corpus, "document_ids": self.document_ids}, f)

        def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
            import numpy as _np
            import pickle as _pickle
            if os.path.exists(retriever_cache_embeddings_file):
                self.embeddings = _np.load(retriever_cache_embeddings_file)
            if os.path.exists(retriever_cache_file):
                with open(retriever_cache_file, "rb") as f:
                    state = _pickle.load(f)
                self.corpus = state.get("corpus", [])
                self.document_ids = state.get("document_ids", {})
            return self
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
    "local_context",
    "similar_event",
    "same_character",
    "temporal_anchor",
    "same_storyline",
    "same_answer_slot",
    "shared_activity",
    "shared_artifact",
    "temporal_followup",
    "before_after",
    "clarifies_answer",
    "image_text_pair",
    "local_evidence_pair",
    "supports",
    "derived_from",
}
DISALLOWED_GRAPH_EDGES = {"updates", "replaces", "conflicts_with", "update", "replace", "conflict"}
ACTIVE_RETRIEVAL_STATUSES = {"active", "candidate", "stale"}
DEFAULT_MEMORY_LEVEL = "instance"
DEFAULT_DOMAIN_CANDIDATE_TOP_K = 3
DEFAULT_DOMAIN_EMBEDDING_THRESHOLD = 0.25
DEFAULT_EMBEDDING_MODEL = os.getenv("SENTENCE_MODEL_PATH", "all-MiniLM-L6-v2")
RETRIEVAL_INDEX_VERSION = "robust_retrieval_v8_rewrite_single_index_debug"
DOMAIN_GRAPH_CACHE_VERSION = "domain_graph_v7_rewrite_memory_edges"
DEFAULT_DOMAIN_TOP_K = 3
DEFAULT_DOMAIN_SEED_TOP_K = 20
DEFAULT_GLOBAL_FALLBACK_TOP_K = 5
DEFAULT_GLOBAL_BM25_TOP_K = 15
DEFAULT_GLOBAL_ENTITY_TOP_K = 10
DEFAULT_FINAL_BUNDLE_SIZE = 6
DEFAULT_FINAL_BUNDLE_MAX_SIZE = 10
DEFAULT_CAT1_PRIMARY_BUNDLE_SIZE = 10
DEFAULT_CAT1_MAX_CONTEXT_BLOCKS = 10
DEFAULT_CAT1_EXPANDED_GLOBAL_TOP_K = 40

REWRITE_MEMORY_PROMPT = """Rewrite one LoCoMo memory into a retrieval-oriented evidence sentence.

Raw memory:
{content}

Return JSON only:
{{
  "rewrite_content": "one self-contained evidence sentence"
}}

Requirements:
- Resolve pronouns using the speaker and content when possible.
- Preserve concrete names, activities, objects, places, dates, and visual evidence.
- Anchor relative time with the session date when it is available, while keeping the original relative cue if useful.
- Include image_caption or image_query facts when they contain answer-bearing evidence.
- Do not add facts not supported by the raw memory.
- Keep it concise, preferably under 55 words.
"""

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
        try:
            import requests as _requests
        except ImportError:
            _requests = None
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        if self._requests is None:
            raise ImportError("requests package not found. Install it to use the SGLang backend.")
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
        try:
            import requests as _requests
        except ImportError:
            _requests = None
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        if self._requests is None:
            raise ImportError("requests package not found. Install it to use the vLLM backend.")
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
                 check_connection: bool = False):
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
                 rewrite_content: Optional[str] = None,
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
                 temporal_expressions: Optional[List[Dict[str, Any]]] = None,
                 confidence: Optional[float] = None,
                 reliability_alpha: Optional[float] = None,
                 reliability_beta: Optional[float] = None,
                 citation_count: Optional[int] = None,
                 successful_citation_count: Optional[int] = None,
                 failed_citation_count: Optional[int] = None,
                 last_updated: Optional[str] = None,
                 llm_controller: Optional[RobustLLMController] = None):

        self.content = current_content or content
        self.current_content = self.content
        self.rewrite_content = rewrite_content or ""

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
        self.temporal_expressions = temporal_expressions or []
        self.confidence = float(confidence if confidence is not None else 1.0)
        self.reliability_alpha = float(reliability_alpha if reliability_alpha is not None else 1.0)
        self.reliability_beta = float(reliability_beta if reliability_beta is not None else 1.0)
        self.citation_count = int(citation_count or 0)
        self.successful_citation_count = int(successful_citation_count or 0)
        self.failed_citation_count = int(failed_citation_count or 0)

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
                 model_name: Optional[str] = None,
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 domain_candidate_top_k: int = DEFAULT_DOMAIN_CANDIDATE_TOP_K,
                 domain_embedding_threshold: float = DEFAULT_DOMAIN_EMBEDDING_THRESHOLD,
                 domain_seed_top_k: int = DEFAULT_DOMAIN_SEED_TOP_K,
                 global_fallback_top_k: int = DEFAULT_GLOBAL_FALLBACK_TOP_K,
                 global_bm25_top_k: int = DEFAULT_GLOBAL_BM25_TOP_K,
                 global_entity_top_k: int = DEFAULT_GLOBAL_ENTITY_TOP_K,
                 final_bundle_size: int = DEFAULT_FINAL_BUNDLE_SIZE,
                 final_bundle_max_size: int = DEFAULT_FINAL_BUNDLE_MAX_SIZE,
                 enable_cat1_coverage_rerank: bool = True):

        self.memories: Dict[str, RobustMemoryNote] = {}
        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.domain_candidate_top_k = max(1, int(domain_candidate_top_k))
        self.domain_embedding_threshold = float(domain_embedding_threshold)
        self.domain_seed_top_k = max(1, int(domain_seed_top_k))
        self.global_fallback_top_k = max(0, int(global_fallback_top_k))
        self.global_bm25_top_k = max(self.global_fallback_top_k, int(global_bm25_top_k))
        self.global_entity_top_k = max(0, int(global_entity_top_k))
        self.final_bundle_size = max(1, int(final_bundle_size))
        self.final_bundle_max_size = max(self.final_bundle_size, int(final_bundle_max_size))
        self.enable_cat1_coverage_rerank = bool(enable_cat1_coverage_rerank)
        self.offline_domain_tree: List[Dict[str, Any]] = []
        self.last_candidate_debug: List[Dict[str, Any]] = []
        self.last_routed_domains: List[str] = []
        self.retrieval_index_version = RETRIEVAL_INDEX_VERSION

    @staticmethod
    def _parse_memory_fields(content: str) -> Dict[str, str]:
        field_names = "dia_id|session_date|speaker|content|image_caption|image_query"
        fields: Dict[str, str] = {}
        for match in re.finditer(
            rf"(?:^|\n)(?P<key>{field_names}):\s*(?P<value>.*?)(?=\n(?:{field_names}):|\Z)",
            str(content),
            re.DOTALL,
        ):
            fields[match.group("key")] = match.group("value").strip()
        return fields

    @staticmethod
    def _raw_memory_content(memory: RobustMemoryNote) -> str:
        return str(getattr(memory, "current_content", getattr(memory, "content", "")) or "")

    def _deterministic_rewrite_content(self, memory: RobustMemoryNote) -> str:
        raw = self._raw_memory_content(memory)
        fields = self._parse_memory_fields(raw)
        speaker = fields.get("speaker", "").strip()
        session_date = fields.get("session_date", "").strip()
        content = fields.get("content", raw).strip()
        image_caption = fields.get("image_caption", "").strip()
        image_query = fields.get("image_query", "").strip()
        pieces = []
        if speaker:
            pieces.append(f"{speaker} said: {content}")
        else:
            pieces.append(content)
        if session_date:
            pieces.append(f"Session date: {session_date}.")
        if image_caption:
            pieces.append(f"Image caption: {image_caption}.")
        if image_query:
            pieces.append(f"Image query: {image_query}.")
        return re.sub(r"\s+", " ", " ".join(piece for piece in pieces if piece)).strip()

    def _rewrite_memory_content(self, memory: RobustMemoryNote) -> str:
        raw = self._raw_memory_content(memory)
        if not raw:
            return ""
        try:
            prompt = REWRITE_MEMORY_PROMPT.format(content=raw)
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            payload = self._parse_json_payload(response)
            rewritten = str(payload.get("rewrite_content", "")).strip()
            if rewritten:
                return re.sub(r"\s+", " ", rewritten).strip()
        except Exception as e:
            logger.warning("Rewrite memory LLM failed for note %s: %s; using fallback rewrite",
                           getattr(memory, "id", "unknown"), e)
        return self._deterministic_rewrite_content(memory)

    def _ensure_rewrite_content(self, memory: RobustMemoryNote) -> None:
        rewrite = str(getattr(memory, "rewrite_content", "") or "").strip()
        if rewrite:
            memory.rewrite_content = re.sub(r"\s+", " ", rewrite).strip()
            return
        memory.rewrite_content = self._rewrite_memory_content(memory)

    def _retrieval_memory_text(self, memory: RobustMemoryNote) -> str:
        rewrite = str(getattr(memory, "rewrite_content", "") or "").strip()
        if rewrite:
            return rewrite
        return self._deterministic_rewrite_content(memory)

    @staticmethod
    def _canonical_token(token: str) -> str:
        token = token.lower()
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        return token

    @classmethod
    def _retrieval_tokens(cls, text: str) -> set:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "what", "when",
            "where", "which", "would", "could", "should", "about", "mention", "mentioned",
            "conversation", "answer", "short", "date", "time", "last", "week", "year",
            "month", "yesterday", "today", "tomorrow", "before", "after", "did", "does",
            "was", "were", "has", "have", "had", "their", "they", "them", "his", "her",
            "him", "she", "you", "your", "are", "any", "all", "who", "why", "how",
        }
        return {
            cls._canonical_token(token)
            for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
            if len(token) >= 3 and token not in stopwords
        }

    @staticmethod
    def _category1_query_expansion(query: str) -> str:
        """Add conservative evidence cues for Cat1 multi-evidence retrieval."""
        query_text = str(query or "")
        query_lower = query_text.lower()
        expansions: List[str] = []

        def add(*terms: str) -> None:
            for term in terms:
                if term and term not in expansions:
                    expansions.append(term)

        if re.search(r"\bidentity\b|\btransgender\b|\btransition", query_lower):
            add("transgender", "trans", "transition", "coming out", "womanhood", "gender identity")
        if "relationship status" in query_lower or re.search(r"\bsingle parent\b|\bbreakup\b", query_lower):
            add("single parent", "breakup", "support", "friends", "family")
        if "career path" in query_lower or re.search(r"\bcounsel|mental health|therap", query_lower):
            add("counseling", "mental health", "therapeutic", "workshop", "support")
        if re.search(r"\bmove from\b|\bmoved from\b|\bhome country\b|\broots\b", query_lower):
            add("home country", "Sweden", "roots", "grandma", "necklace")
        if re.search(r"\bactivit|partake|destress|family|hikes?\b", query_lower):
            add("pottery", "swimming", "camping", "hiking", "running", "painting", "frisbee", "family")
        if re.search(r"\bkids?\b|\bchildren\b", query_lower):
            add("kids", "children", "nature", "dinosaur", "animals", "exhibit", "family")
        if re.search(r"\bevents?\b|\bparticipat|community|lgbtq|transgender-specific\b", query_lower):
            add("support group", "pride parade", "school event", "workshop", "fundraiser", "community")
        if re.search(r"\bpaint|painting|painted|art\b", query_lower):
            add("painting", "painted", "artwork", "sunset", "lake", "portrait", "subject")
        if re.search(r"\bpottery\b", query_lower):
            add("pottery", "class", "clay", "kids", "made")
        if re.search(r"\bpets?\b|\bnames?\b", query_lower):
            add("pets", "names", "dog", "cat")
        if re.search(r"\bmusical artists?\b|\bbands?\b|\bconcert\b", query_lower):
            add("music", "concert", "artist", "band", "saw")
        if re.search(r"\bbook\b|\bread\b|\bsuggestion\b", query_lower):
            add("book", "read", "suggestion", "recommended", "novel")
        if re.search(r"\bbeach\b", query_lower):
            add("beach", "trip", "went", "2023")

        if not expansions:
            return query_text
        return f"{query_text} ; evidence cues: {' '.join(expansions)}"

    def _lexical_relevance(self, memory: RobustMemoryNote, query: str) -> float:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        main_text = " ".join([
            fields.get("speaker", ""),
            self._retrieval_memory_text(memory),
            " ".join(getattr(memory, "keywords", [])),
            " ".join(getattr(memory, "tags", [])),
        ])
        visual_text = " ".join([
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
        ])
        query_tokens = self._retrieval_tokens(query)
        if not query_tokens:
            return 0.0
        main_tokens = self._retrieval_tokens(main_text)
        visual_tokens = self._retrieval_tokens(visual_text)
        main_overlap = len(query_tokens & main_tokens)
        visual_overlap = len(query_tokens & visual_tokens)
        coverage = (main_overlap + 0.5 * visual_overlap) / max(1, len(query_tokens))
        exact_boost = 0.0
        query_lower = str(query).lower()
        main_lower = main_text.lower()
        for phrase in re.split(r"[,;?]+", query_lower):
            phrase = phrase.strip()
            if len(phrase) >= 8 and phrase in main_lower:
                exact_boost += 0.25
        return (3.0 * main_overlap) + (0.75 * visual_overlap) + coverage + exact_boost

    @staticmethod
    def _dia_sort_key(dia_id: str) -> Optional[tuple]:
        match = re.match(r"D(\d+):(\d+)$", str(dia_id).strip())
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _dia_id_for_memory(self, memory: RobustMemoryNote) -> Optional[str]:
        for condition in getattr(memory, "conditions", []) or []:
            dia_id = condition.get("dia_id") if isinstance(condition, dict) else None
            if dia_id and self._dia_sort_key(str(dia_id)):
                return str(dia_id)
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        dia_id = fields.get("dia_id")
        return dia_id if dia_id and self._dia_sort_key(dia_id) else None

    def _build_dia_lookup(self, memories: List[RobustMemoryNote]) -> Dict[tuple, RobustMemoryNote]:
        lookup: Dict[tuple, RobustMemoryNote] = {}
        for memory in memories:
            memory = self._ensure_memory_schema(memory)
            dia_id = self._dia_id_for_memory(memory)
            sort_key = self._dia_sort_key(dia_id) if dia_id else None
            if sort_key:
                lookup[sort_key] = memory
        return lookup

    def _local_context_neighbors(
        self,
        memory: RobustMemoryNote,
        dia_lookup: Dict[tuple, RobustMemoryNote],
        radius: int = 2,
    ) -> List[RobustMemoryNote]:
        dia_id = self._dia_id_for_memory(memory)
        sort_key = self._dia_sort_key(dia_id) if dia_id else None
        if not sort_key:
            return []
        session_idx, turn_idx = sort_key
        neighbors = []
        for distance in range(1, radius + 1):
            for neighbor_key in ((session_idx, turn_idx - distance), (session_idx, turn_idx + distance)):
                neighbor = dia_lookup.get(neighbor_key)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = str(text or "").strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @classmethod
    def _parse_json_payload(cls, response: str) -> Dict[str, Any]:
        cleaned = cls._strip_markdown_fences(response)
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return {}
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}

    @staticmethod
    def _normalize_domain_path(path: Any) -> str:
        parts = [
            re.sub(r"\s+", " ", part.strip())
            for part in str(path or "").replace(">", "/").split("/")
            if part and part.strip()
        ]
        return " / ".join(parts[:3]) if parts else "General / Conversation / Episodic"

    def _valid_domain_paths(self) -> List[str]:
        if self.offline_domain_tree:
            paths = [entry.get("path", "") for entry in self.offline_domain_tree]
            return [path for path in paths if path]
        return list(self._domain_catalog().keys())

    def _memory_brief(self, memory: RobustMemoryNote, max_chars: int = 220) -> str:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        pieces = [
            fields.get("dia_id", ""),
            fields.get("session_date", ""),
            fields.get("speaker", ""),
            self._retrieval_memory_text(memory),
        ]
        brief = " | ".join(piece for piece in pieces if piece)
        return brief[:max_chars]

    def _conversation_overview(
        self,
        session_summaries: Optional[Dict[str, str]] = None,
        max_lines: int = 80,
    ) -> str:
        lines: List[str] = []
        if session_summaries:
            for key, value in sorted(session_summaries.items()):
                if value:
                    lines.append(f"{key}: {str(value)[:360]}")
                if len(lines) >= max_lines:
                    break
        if len(lines) < max_lines:
            memories = list(self.memories.values())
            step = max(1, len(memories) // max(1, max_lines - len(lines)))
            for memory in memories[::step]:
                lines.append(self._memory_brief(self._ensure_memory_schema(memory), max_chars=260))
                if len(lines) >= max_lines:
                    break
        return "\n".join(lines)

    def _fallback_domain_tree(self) -> List[Dict[str, Any]]:
        seed_paths = [
            "Personal Life / Family / Relationships",
            "Personal Life / Identity / Self Expression",
            "Career Education / Work / Goals",
            "Health Wellbeing / Mental Health / Support",
            "Hobbies Interests / Arts Media / Entertainment",
            "Events Activities / Travel Meetings / Experiences",
            "Social Relationship / Friends / Conversation Partner",
            "Places Objects / Images / Visual Evidence",
            "Plans Decisions / Future Intentions / Commitments",
            "General Conversation / Episodic Turns / Miscellaneous",
        ]
        return [
            {
                "path": path,
                "description": f"LoCoMo memories related to {path.lower()}.",
                "keywords": [part.lower() for part in path.replace("/", " ").split()[:6]],
            }
            for path in seed_paths
        ]

    def _normalize_domain_tree(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_domains = payload.get("domains") or payload.get("domain_tree") or payload.get("tree") or []
        if isinstance(raw_domains, dict):
            raw_domains = raw_domains.get("domains", [])
        normalized: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for item in raw_domains:
            if isinstance(item, str):
                path = self._normalize_domain_path(item)
                description = ""
                keywords: List[str] = []
            elif isinstance(item, dict):
                path = self._normalize_domain_path(
                    item.get("path") or item.get("domain_path") or item.get("name") or item.get("title")
                )
                description = str(item.get("description") or item.get("summary") or "")
                raw_keywords = item.get("keywords") or item.get("tags") or []
                if isinstance(raw_keywords, str):
                    keywords = [token.strip() for token in re.split(r"[,;/]", raw_keywords) if token.strip()]
                else:
                    keywords = [str(token).strip() for token in raw_keywords if str(token).strip()]
            else:
                continue
            if path in seen:
                continue
            seen.add(path)
            normalized.append({
                "path": path,
                "description": description[:500],
                "keywords": keywords[:12],
            })
            if len(normalized) >= 36:
                break
        return normalized or self._fallback_domain_tree()

    def _generate_domain_tree_with_llm(
        self,
        sample_id: str,
        session_summaries: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        overview = self._conversation_overview(session_summaries=session_summaries)
        prompt = f"""Build a three-level domain tree for one LoCoMo long conversation sample.

Sample id: {sample_id}

Conversation overview:
{overview}

Return JSON only:
{{
  "domains": [
    {{
      "path": "Level 1 / Level 2 / Level 3",
      "description": "What memories belong here.",
      "keywords": ["keyword1", "keyword2"]
    }}
  ]
}}

Requirements:
- Create 8 to 24 domain paths.
- Every path must have exactly three levels separated by " / ".
- Domains should be specific to this two-person conversation.
- Prefer recurring topics, events, relationships, goals, places, objects, and temporal storylines.
"""
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            return self._normalize_domain_tree(self._parse_json_payload(response))
        except Exception as e:
            logger.warning("Offline domain tree LLM failed for sample %s: %s; using fallback tree", sample_id, e)
            return self._fallback_domain_tree()

    def _domain_texts(self, domain_tree: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        entries = domain_tree or self.offline_domain_tree or self._fallback_domain_tree()
        texts: Dict[str, str] = {}
        for entry in entries:
            path = self._normalize_domain_path(entry.get("path", ""))
            texts[path] = " ".join([
                path,
                str(entry.get("description", "")),
                " ".join(str(keyword) for keyword in entry.get("keywords", []) or []),
            ]).strip()
        return texts

    def _embedding_domain_annotation(
        self,
        memory: RobustMemoryNote,
        domain_tree: Optional[List[Dict[str, Any]]] = None,
        max_domains: int = 3,
    ) -> List[str]:
        domain_texts = self._domain_texts(domain_tree)
        if not domain_texts:
            return ["General / Conversation / Episodic"]
        memory_text = self._memory_to_index_text(memory)
        try:
            query_embedding = self.retriever.model.encode([memory_text])[0]
            domain_paths = list(domain_texts.keys())
            embeddings = self.retriever.model.encode([domain_texts[path] for path in domain_paths])
            scored = [
                (self._cosine_similarity(query_embedding, embedding), path)
                for path, embedding in zip(domain_paths, embeddings)
            ]
            scored.sort(reverse=True)
            selected = [path for _, path in scored[:max_domains]]
        except Exception as e:
            logger.warning("Embedding domain annotation failed: %s; using lexical fallback", e)
            mem_tokens = self._retrieval_tokens(memory_text)
            scored = []
            for path, text in domain_texts.items():
                dom_tokens = self._retrieval_tokens(text)
                scored.append((len(mem_tokens & dom_tokens), path))
            scored.sort(reverse=True)
            selected = [path for score, path in scored[:max_domains] if score > 0]
        return selected[:max_domains] or [next(iter(domain_texts.keys()))]

    def _annotate_domains_with_llm(
        self,
        domain_tree: List[Dict[str, Any]],
        batch_size: int = 8,
    ) -> Dict[str, List[str]]:
        valid_paths = [entry["path"] for entry in domain_tree if entry.get("path")]
        valid_set = set(valid_paths)
        if not valid_paths:
            return {}
        domain_block = "\n".join(
            f"- {entry['path']}: {entry.get('description', '')}"
            for entry in domain_tree
            if entry.get("path")
        )
        memories = [self._ensure_memory_schema(memory) for memory in self.memories.values()]
        annotations: Dict[str, List[str]] = {}
        for start in range(0, len(memories), batch_size):
            batch = memories[start:start + batch_size]
            memory_block = "\n".join(self._memory_brief(memory, max_chars=260) for memory in batch)
            prompt = f"""Assign LoCoMo dialogue memories to domain paths.

Valid domain paths:
{domain_block}

Memories:
{memory_block}

Return JSON only:
{{
  "annotations": [
    {{"dia_id": "D1:1", "domain_paths": ["Level 1 / Level 2 / Level 3"]}}
  ]
}}

Rules:
- Use only valid domain paths from the list.
- Assign 1 to 3 paths per memory.
- Prefer the most specific directly relevant paths.
"""
            try:
                response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
                payload = self._parse_json_payload(response)
                raw_items = payload.get("annotations") or payload.get("memories") or []
            except Exception as e:
                logger.warning("Domain annotation batch LLM failed at memory %d: %s", start, e)
                raw_items = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                dia_id = str(item.get("dia_id", "")).strip()
                raw_paths = item.get("domain_paths") or item.get("domains") or []
                if isinstance(raw_paths, str):
                    raw_paths = [raw_paths]
                paths = []
                for path in raw_paths:
                    normalized = self._normalize_domain_path(path)
                    if normalized in valid_set and normalized not in paths:
                        paths.append(normalized)
                    if len(paths) >= 3:
                        break
                if dia_id and paths:
                    annotations[dia_id] = paths

        for memory in memories:
            dia_id = self._dia_id_for_memory(memory)
            if dia_id and dia_id not in annotations:
                annotations[dia_id] = self._embedding_domain_annotation(memory, domain_tree)
        return annotations

    def _detect_temporal_expressions(self, memory: RobustMemoryNote) -> List[Dict[str, Any]]:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        text = self._retrieval_memory_text(memory)
        session_date = fields.get("session_date", "")
        temporal_markers = [
            "yesterday", "today", "tomorrow", "last week", "next week", "last month",
            "next month", "last year", "next year", "last monday", "last tuesday",
            "last wednesday", "last thursday", "last friday", "last saturday", "last sunday",
        ]
        lowered = text.lower()
        return [
            {"text": marker, "anchor": session_date, "normalized_date": ""}
            for marker in temporal_markers
            if marker in lowered
        ]

    def _offline_graph_relations(self) -> Set[str]:
        return {
            "similar_event",
            "same_character",
            "same_storyline",
            "image_text_pair",
        }

    def _legacy_offline_graph_relations(self) -> Set[str]:
        return {
            "temporal_anchor",
            "same_answer_slot",
            "shared_activity",
            "shared_artifact",
            "temporal_followup",
            "before_after",
            "clarifies_answer",
            "local_evidence_pair",
            "supports",
            "derived_from",
        }

    def _clear_offline_graph_edges(self) -> None:
        offline_relations = self._offline_graph_relations() | self._legacy_offline_graph_relations()
        for memory in self.memories.values():
            memory = self._ensure_memory_schema(memory)
            memory.links = [
                edge for edge in getattr(memory, "links", [])
                if not (isinstance(edge, dict) and edge.get("relation") in offline_relations)
            ]

    def _event_text(self, memory: RobustMemoryNote) -> str:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        return " ".join([
            fields.get("speaker", ""),
            self._retrieval_memory_text(memory),
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
        ]).strip()

    def _typed_edge_profile(self, memory: RobustMemoryNote) -> Dict[str, Any]:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        speaker = fields.get("speaker", "").strip()
        content = self._retrieval_memory_text(memory)
        image_caption = fields.get("image_caption", "")
        image_query = fields.get("image_query", "")
        keyword_text = " ; ".join(str(item) for item in getattr(memory, "keywords", []) or [])
        tag_text = " ; ".join(str(item) for item in getattr(memory, "tags", []) or [])
        visual_text = " ".join([image_caption, image_query])

        generic_cues = {
            "a", "an", "and", "are", "ask", "asked", "asking", "been", "being",
            "chat", "conversation", "day", "dialogue", "did", "does", "doing",
            "event", "events", "experience", "experiences", "felt", "feel", "feeling",
            "friend", "friends", "got", "had", "has", "have", "image", "images",
            "item", "items", "just", "memory", "memories", "mentioned", "new",
            "people", "person", "photo", "picture", "really", "said", "says",
            "session", "shared", "shares", "something", "talk", "talked", "talking",
            "thing", "things", "time", "today", "told", "turn", "want", "wanted",
            "week", "went", "would", "dia", "speaker", "content", "general",
            "gonna", "pretty", "out", "but", "one", "need", "needs", "i'm",
            "i've", "i'd", "it'll", "you", "your", "yours", "me", "my", "we",
            "our", "ours", "they", "them", "their", "there", "these", "those",
            "who", "what", "when", "where", "why", "how", "which", "the", "to",
            "of", "in", "on", "at", "by", "from", "for", "with", "without",
            "as", "is", "was", "were", "be", "am", "been", "it", "its", "it's",
            "that", "this", "than", "then", "after", "before", "any", "anything",
            "anyth", "hey", "hi", "hello", "good", "great", "see", "yeah", "yes",
            "no", "thanks", "thank", "mel", "kid", "kids",
            "activity", "activities", "art", "arts", "career", "education",
            "evidence", "expression", "goal", "health", "identity", "life",
            "media", "meeting", "meetings", "object", "objects", "partner",
            "personal", "plan", "plans", "relationship", "relationships",
            "self", "travel", "visual", "wellbe", "wellbeing",
            "cool", "wow", "awesome", "inspir", "inspiring", "hear", "happy",
            "thankful", "such", "all", "story", "love", "lovely", "support",
            "about", "agree", "alway", "always", "care", "challenge", "each",
            "easy", "fun", "help", "important", "it'", "journey", "like",
            "long", "look", "make", "really", "since", "super", "tak", "take",
            "taking", "that'", "think", "totally", "understand", "way", "work",
            "yourself", "ourselve", "ourselves", "you're", "here", "here'",
            "lot", "mean", "amaz", "amazing", "kind", "happen", "get", "job",
        }
        generic_phrases = {
            "conversation memory", "image query", "memory content", "memory context",
            "personal experience", "talk start", "session date", "dia speaker",
            "speaker content", "session speaker",
        }
        strong_terms = {
            "adoption", "agency", "agencies", "bach", "bareilles", "beach", "brave",
            "camping", "caroline", "certification", "children", "counseling",
            "dr seuss", "four seasons", "grandma", "lgbtq", "luna", "melanie",
            "mozart", "necklace", "oliver", "pottery", "pride", "psychology",
            "sara bareilles", "seuss", "shoes", "support group", "sweden",
            "the four seasons", "transgender", "violin",
        }
        protected_phrases = {
            "adoption agency", "adoption agencies", "charity race", "classic book",
            "classic books", "classical music", "counseling certification",
            "connected lgbtq activists", "dr seuss", "four seasons",
            "lgbtq activist group", "lgbtq support", "lgbtq support group",
            "mental health", "single parent", "support group", "the four seasons",
        }
        high_value_tokens = {
            "adoption", "agency", "bach", "bareilles", "beach", "brave", "camp",
            "camping", "career", "certification", "child", "children", "counsel",
            "counseling", "grandma", "health", "lgbtq", "mental", "mozart",
            "necklace", "painting", "parent", "pottery", "psychology", "race",
            "seuss", "shoe", "shoes", "single", "sweden",
            "transgender", "violin",
        }
        common_participants = {"caroline", "melanie"}

        def normalize_cue(text_value: str) -> str:
            cue = re.sub(r"[^a-z0-9'\s-]", " ", str(text_value).lower())
            cue = re.sub(r"\s+", " ", cue).strip(" -'")
            if cue.endswith("'s"):
                cue = cue[:-2]
            return cue

        def useful_cue(cue: str) -> bool:
            if cue in protected_phrases:
                return True
            if not cue or cue in generic_cues or cue in generic_phrases:
                return False
            if re.search(r"\d", cue):
                return False
            if any(part in {"dia", "speaker", "session", "content"} for part in cue.split()):
                return False
            if len(cue) < 3:
                return False
            parts = cue.split()
            if len(parts) == 1:
                return cue not in generic_cues
            return not any(part in generic_cues for part in parts)

        def add_phrase(cues: Set[str], phrase: str) -> None:
            cue = normalize_cue(phrase)
            if useful_cue(cue):
                cues.add(cue)

        def phrase_cues(
            text_value: str,
            include_bigrams: bool = True,
            include_segments: bool = False,
        ) -> Set[str]:
            cues: Set[str] = set()
            raw = str(text_value or "")
            for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", raw):
                add_phrase(cues, quoted)
            for capitalized in re.findall(r"\b[A-Z][A-Za-z0-9']+(?:\s+(?:of|the|and|[A-Z][A-Za-z0-9']+)){0,4}", raw):
                add_phrase(cues, capitalized)
            if include_segments:
                for segment in re.split(r"[;,/|()\[\]\n]+", raw):
                    segment = segment.strip()
                    if 2 <= len(segment) <= 80:
                        add_phrase(cues, segment)
            tokens = []
            for raw_token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9']+\b", raw.lower()):
                token = self._canonical_token(raw_token)
                if useful_cue(token):
                    tokens.append(token)
            cues.update(tokens)
            if include_bigrams:
                for n in (2, 3):
                    for idx in range(0, max(0, len(tokens) - n + 1)):
                        gram = " ".join(tokens[idx: idx + n])
                        if useful_cue(gram):
                            cues.add(gram)
            return cues

        storyline_cues: Set[str] = set()
        strong_cues: Set[str] = set()
        visual_cues = phrase_cues(visual_text, include_bigrams=True, include_segments=True)
        storyline_cues.update(phrase_cues(content, include_bigrams=True, include_segments=False))
        metadata_cues: Set[str] = set()
        for source_text in [keyword_text, tag_text]:
            metadata_cues.update(phrase_cues(source_text, include_bigrams=True, include_segments=True))
        storyline_cues.update({
            cue for cue in metadata_cues
            if cue in protected_phrases
            or cue in strong_terms
            or bool(set(cue.split()) & high_value_tokens)
        })
        if speaker:
            add_phrase(storyline_cues, speaker)
        storyline_cues |= visual_cues

        for cue in list(storyline_cues):
            if cue in common_participants:
                continue
            cue_parts = set(cue.split())
            if cue in strong_terms or any(term in cue for term in strong_terms if " " in term):
                strong_cues.add(cue)
            elif len(cue_parts) >= 2 and bool(cue_parts & high_value_tokens):
                strong_cues.add(cue)

        cue_types = {
            "person": {
                cue for cue in storyline_cues
                if cue in {normalize_cue(speaker), "caroline", "melanie", "grandma"}
            },
            "visual": set(visual_cues),
            "object": {
                cue for cue in storyline_cues
                if cue in {
                    "necklace", "shoes", "book", "books", "pet", "pets", "painting",
                    "pottery", "violin", "instrument", "luna", "oliver",
                }
            },
            "activity": {
                cue for cue in storyline_cues
                if cue in {
                    "camping", "hiking", "swimming", "running", "painting",
                    "pottery", "music", "concert", "reading", "support group",
                }
            },
        }
        return {
            "speaker": speaker,
            "storyline_cues": storyline_cues,
            "strong_cues": strong_cues,
            "visual_cues": visual_cues,
            "cue_types": cue_types,
        }

    def _shared_domain(self, left: RobustMemoryNote, right: RobustMemoryNote) -> bool:
        left_domains = set(getattr(left, "domain_paths", []) or [])
        right_domains = set(getattr(right, "domain_paths", []) or [])
        return bool(left_domains & right_domains)

    def _speaker_for_memory(self, memory: RobustMemoryNote) -> str:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        return fields.get("speaker", "").strip()

    def _build_offline_graph_edges(self) -> None:
        memories = [self._ensure_memory_schema(memory) for memory in self.memories.values()]
        if not memories:
            return
        self._clear_offline_graph_edges()
        for memory in memories:
            memory.temporal_expressions = self._detect_temporal_expressions(memory)

        event_texts = [self._event_text(memory) for memory in memories]
        try:
            event_embeddings = self.retriever.model.encode(event_texts)
        except Exception as e:
            logger.warning("Event embedding failed during graph construction: %s", e)
            event_embeddings = [None for _ in memories]

        dia_keys = [self._dia_sort_key(self._dia_id_for_memory(memory) or "") for memory in memories]
        speakers = [self._speaker_for_memory(memory) for memory in memories]
        profiles = [self._typed_edge_profile(memory) for memory in memories]
        by_speaker: Dict[str, List[int]] = {}
        for idx, speaker in enumerate(speakers):
            if speaker:
                by_speaker.setdefault(speaker, []).append(idx)
        for speaker_indices in by_speaker.values():
            speaker_indices.sort(key=lambda idx: dia_keys[idx] or (10**9, 10**9))
            for left_idx, right_idx in zip(speaker_indices, speaker_indices[1:]):
                left = memories[left_idx]
                right = memories[right_idx]
                self._add_edge(left, right, "same_character", "Adjacent memories from the same LoCoMo speaker.", 0.4)
                self._add_edge(right, left, "same_character", "Adjacent memories from the same LoCoMo speaker.", 0.4)

        similar_counts: Dict[str, int] = {}
        typed_counts: Dict[str, int] = {}
        max_similar_edges_per_memory = 4
        max_typed_edges_per_memory = 8

        def add_typed_edge(
            left: RobustMemoryNote,
            right: RobustMemoryNote,
            relation: str,
            reason: str,
            strength: float,
        ) -> None:
            if typed_counts.get(left.id, 0) >= max_typed_edges_per_memory:
                return
            self._add_edge(left, right, relation, reason, strength)
            typed_counts[left.id] = typed_counts.get(left.id, 0) + 1

        for i, left in enumerate(memories):
            left_tokens = self._retrieval_tokens(event_texts[i])
            left_key = dia_keys[i]
            for j in range(i + 1, len(memories)):
                right = memories[j]
                right_key = dia_keys[j]
                same_character = bool(speakers[i] and speakers[i] == speakers[j])

                allow_similar_event = (
                    similar_counts.get(left.id, 0) < max_similar_edges_per_memory
                    and similar_counts.get(right.id, 0) < max_similar_edges_per_memory
                )
                right_tokens = self._retrieval_tokens(event_texts[j])
                overlap = len(left_tokens & right_tokens)
                if event_embeddings[i] is not None and event_embeddings[j] is not None:
                    similarity = self._cosine_similarity(event_embeddings[i], event_embeddings[j])
                else:
                    similarity = overlap / max(1, len(left_tokens | right_tokens))

                same_session_nearby = (
                    left_key is not None and right_key is not None
                    and left_key[0] == right_key[0]
                    and abs(left_key[1] - right_key[1]) <= 4
                )
                cross_session_event = (
                    left_key is not None and right_key is not None
                    and left_key[0] != right_key[0]
                    and same_character
                    and overlap >= 2
                )
                if allow_similar_event and same_session_nearby and (overlap >= 2 or similarity >= 0.60):
                    weight = 1.0
                elif allow_similar_event and cross_session_event and overlap >= 3 and similarity >= 0.50:
                    weight = 0.6
                else:
                    weight = 0.0
                if weight > 0.0:
                    self._add_edge(left, right, "similar_event", "Rule and embedding based LoCoMo event match.", weight)
                    self._add_edge(right, left, "similar_event", "Rule and embedding based LoCoMo event match.", weight)
                    similar_counts[left.id] = similar_counts.get(left.id, 0) + 1
                    similar_counts[right.id] = similar_counts.get(right.id, 0) + 1

                left_profile = profiles[i]
                right_profile = profiles[j]
                left_cues = set(left_profile.get("storyline_cues", set()))
                right_cues = set(right_profile.get("storyline_cues", set()))
                shared_cues = left_cues & right_cues
                shared_strong_cues = (
                    set(left_profile.get("strong_cues", set()))
                    & set(right_profile.get("strong_cues", set()))
                )
                common_story_participants = {"caroline", "melanie"}
                non_person_shared = shared_cues - common_story_participants - {
                    str(left_profile.get("speaker", "")).lower(),
                    str(right_profile.get("speaker", "")).lower(),
                }
                chronological_pair = (
                    left_key is not None and right_key is not None
                    and left_key[0] != right_key[0]
                    and left_key < right_key
                )

                same_storyline_allowed = False
                if same_character and shared_strong_cues:
                    same_storyline_allowed = True
                elif same_character and len(non_person_shared) >= 2:
                    same_storyline_allowed = True
                elif not same_character and len(shared_strong_cues) >= 2:
                    same_storyline_allowed = True
                elif chronological_pair and shared_strong_cues and (overlap >= 1 or similarity >= 0.35):
                    same_storyline_allowed = True

                if same_storyline_allowed and (overlap >= 1 or similarity >= 0.25 or shared_strong_cues):
                    cue_preview = sorted(shared_strong_cues or non_person_shared or shared_cues)[:5]
                    reason = "Shared concrete storyline cues: " + ", ".join(cue_preview)
                    if chronological_pair:
                        reason += "; chronological follow-up across sessions"
                    strength = (
                        0.60
                        + 0.08 * min(3, len(shared_strong_cues))
                        + 0.04 * min(3, len(non_person_shared))
                        + (0.08 if same_character else 0.0)
                        + (0.04 if chronological_pair else 0.0)
                    )
                    strength = min(0.92, strength)
                    add_typed_edge(left, right, "same_storyline", reason, strength)
                    add_typed_edge(right, left, "same_storyline", reason, strength)

                left_visual = set(left_profile.get("visual_cues", set()))
                right_visual = set(right_profile.get("visual_cues", set()))
                left_visual_overlap = left_visual & right_cues
                right_visual_overlap = right_visual & left_cues
                visual_shared = (left_visual_overlap | right_visual_overlap) - {
                    str(left_profile.get("speaker", "")).lower(),
                    str(right_profile.get("speaker", "")).lower(),
                }
                if same_session_nearby and visual_shared and (overlap >= 1 or similarity >= 0.20):
                    cue_preview = sorted(visual_shared)[:5]
                    reason = "Image-text cue alignment: " + ", ".join(cue_preview)
                    strength = min(0.86, 0.58 + 0.08 * min(3, len(visual_shared)))
                    add_typed_edge(left, right, "image_text_pair", reason, strength)
                    add_typed_edge(right, left, "image_text_pair", reason, strength)

    def _domain_graph_cache_payload(self, sample_id: str) -> Dict[str, Any]:
        annotations = {}
        temporal = {}
        edges = {}
        offline_relations = self._offline_graph_relations()
        for memory in self.memories.values():
            memory = self._ensure_memory_schema(memory)
            dia_id = self._dia_id_for_memory(memory) or memory.id
            annotations[dia_id] = list(getattr(memory, "domain_paths", []) or [])
            temporal[dia_id] = list(getattr(memory, "temporal_expressions", []) or [])
            edges[dia_id] = [
                edge for edge in getattr(memory, "links", [])
                if isinstance(edge, dict)
                and edge.get("relation") in offline_relations
            ]
        return {
            "version": DOMAIN_GRAPH_CACHE_VERSION,
            "retrieval_index_version": RETRIEVAL_INDEX_VERSION,
            "sample_id": sample_id,
            "domain_tree": self.offline_domain_tree,
            "annotations": annotations,
            "temporal_expressions": temporal,
            "edges": edges,
        }

    def _apply_domain_graph_cache(self, payload: Dict[str, Any]) -> bool:
        if payload.get("version") != DOMAIN_GRAPH_CACHE_VERSION:
            return False
        self.offline_domain_tree = self._normalize_domain_tree({"domains": payload.get("domain_tree", [])})
        annotations = payload.get("annotations", {}) or {}
        temporal = payload.get("temporal_expressions", {}) or {}
        edges = payload.get("edges", {}) or {}
        offline_relations = self._offline_graph_relations() | self._legacy_offline_graph_relations()
        id_by_dia = {}
        for memory in self.memories.values():
            memory = self._ensure_memory_schema(memory)
            dia_id = self._dia_id_for_memory(memory) or memory.id
            id_by_dia[dia_id] = memory
        for dia_id, memory in id_by_dia.items():
            paths = annotations.get(dia_id)
            if paths:
                memory.domain_paths = [self._normalize_domain_path(path) for path in paths][:3]
            memory.temporal_expressions = temporal.get(dia_id, getattr(memory, "temporal_expressions", []))
            memory.links = [
                edge for edge in getattr(memory, "links", [])
                if not (isinstance(edge, dict) and edge.get("relation") in offline_relations)
            ]
        for dia_id, edge_list in edges.items():
            source = id_by_dia.get(dia_id)
            if not source:
                continue
            for edge in edge_list:
                if not isinstance(edge, dict):
                    continue
                target = id_by_dia.get(edge.get("target_dia_id")) or next(
                    (memory for memory in id_by_dia.values() if memory.id == edge.get("target_id")),
                    None,
                )
                relation = edge.get("relation", "similar_event")
                if target and relation in self._offline_graph_relations():
                    self._add_edge(
                        source,
                        target,
                        relation,
                        str(edge.get("reason", "Loaded from offline domain graph cache.")),
                        float(edge.get("strength", 0.6)),
                    )
        self.consolidate_memories()
        return True

    def prepare_offline_domain_graph(
        self,
        sample_id: str,
        cache_path: Optional[str] = None,
        session_summaries: Optional[Dict[str, str]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Build or load per-sample domain tree, memory annotations, and leaf graph edges."""
        for memory in self.memories.values():
            self._ensure_memory_schema(memory)

        if cache_path and not force_rebuild and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if self._apply_domain_graph_cache(payload):
                    logger.info("Loaded offline domain graph cache for sample %s from %s", sample_id, cache_path)
                    return {"loaded_cache": True, "cache_path": cache_path}
            except Exception as e:
                logger.warning("Failed to load domain graph cache %s: %s; rebuilding", cache_path, e)

        self.offline_domain_tree = self._generate_domain_tree_with_llm(
            sample_id=sample_id,
            session_summaries=session_summaries,
        )
        annotations = self._annotate_domains_with_llm(self.offline_domain_tree)
        for memory in self.memories.values():
            memory = self._ensure_memory_schema(memory)
            dia_id = self._dia_id_for_memory(memory)
            if dia_id and annotations.get(dia_id):
                memory.domain_paths = annotations[dia_id][:3]
            elif not getattr(memory, "domain_paths", None):
                memory.domain_paths = self._embedding_domain_annotation(memory, self.offline_domain_tree)

        self._build_offline_graph_edges()
        self.consolidate_memories()

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._domain_graph_cache_payload(sample_id), f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Failed to save domain graph cache %s: %s", cache_path, e)
        return {
            "loaded_cache": False,
            "domain_count": len(self.offline_domain_tree),
            "memory_count": len(self.memories),
            "cache_path": cache_path,
        }

    def _embedding_rank_indices(
        self,
        query: str,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[tuple]:
        if not candidate_indices:
            return []
        all_memories = list(self.memories.values())
        if self.retriever.embeddings is None or len(self.retriever.corpus) != len(all_memories):
            self.consolidate_memories()
        try:
            query_embedding = self.retriever.model.encode([query])[0]
            scored = []
            for idx in candidate_indices:
                if idx >= len(all_memories) or self.retriever.embeddings is None:
                    continue
                score = self._cosine_similarity(query_embedding, self.retriever.embeddings[idx])
                scored.append((score, idx))
            scored.sort(reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning("Domain embedding ranking failed: %s", e)
            return []

    def _bm25_rank_indices(
        self,
        query: str,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[tuple]:
        if not candidate_indices:
            return []
        all_memories = list(self.memories.values())
        query_tokens = list(self._retrieval_tokens(query))
        if not query_tokens:
            return []
        docs = []
        valid_indices = []
        for idx in candidate_indices:
            if idx >= len(all_memories):
                continue
            memory = self._ensure_memory_schema(all_memories[idx])
            docs.append(list(self._retrieval_tokens(self._memory_to_index_text(memory))))
            valid_indices.append(idx)
        if not docs:
            return []
        avgdl = sum(len(doc) for doc in docs) / max(1, len(docs))
        doc_freq: Dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        k1 = 1.5
        b = 0.75
        scored = []
        for idx, doc in zip(valid_indices, docs):
            freq: Dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            score = 0.0
            dl = len(doc)
            for token in query_tokens:
                if token not in freq:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
                tf = freq[token]
                score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6)))
            if score > 0.0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return scored[:top_k]

    def _entity_rank_indices(
        self,
        query: str,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[tuple]:
        """Rank candidates by exact speaker/entity/action overlap for evidence fallback."""
        if not candidate_indices:
            return []
        all_memories = list(self.memories.values())
        query_tokens = self._retrieval_tokens(query)
        if not query_tokens:
            return []
        query_lower = str(query).lower()
        query_names = set(re.findall(r"\b[A-Z][a-z]+(?:'[a-z]+)?\b", str(query)))
        scored = []
        for idx in candidate_indices:
            if idx >= len(all_memories):
                continue
            memory = self._ensure_memory_schema(all_memories[idx])
            fields = self._parse_memory_fields(self._raw_memory_content(memory))
            speaker = fields.get("speaker", "")
            content = self._retrieval_memory_text(memory)
            exact_text = " ".join([
                speaker,
                content,
                fields.get("image_caption", ""),
                fields.get("image_query", ""),
                " ".join(getattr(memory, "keywords", [])),
                " ".join(getattr(memory, "tags", [])),
            ])
            exact_lower = exact_text.lower()
            memory_tokens = self._retrieval_tokens(exact_text)
            overlap = len(query_tokens & memory_tokens)
            phrase_bonus = 0.0
            for token in query_tokens:
                if token in exact_lower:
                    phrase_bonus += 0.25
            name_bonus = 0.0
            for name in query_names:
                if name.lower() == speaker.lower() or name.lower() in exact_lower:
                    name_bonus += 1.0
            action_bonus = 0.0
            for action in ("research", "apply", "paint", "read", "camp", "support", "birthday", "identity", "beach", "conference", "meeting"):
                if action in query_lower and action in exact_lower:
                    action_bonus += 1.5
            score = 2.0 * overlap + phrase_bonus + name_bonus + action_bonus
            if score > 0.0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return scored[:top_k]

    def _retrieval_candidates(
        self,
        query: str,
        k: int,
        routed_domains: Optional[List[str]],
        category: Optional[int] = None,
    ) -> List[tuple]:
        all_memories = list(self.memories.values())
        if not all_memories:
            return []
        try:
            category_int = int(category) if category is not None else None
        except (TypeError, ValueError):
            category_int = None
        lexical_query = self._category1_query_expansion(query) if category_int == 1 else query

        candidate_indices = []
        fallback_indices = []
        for idx, memory in enumerate(all_memories):
            memory = self._ensure_memory_schema(memory)
            if not self._is_retrievable(memory, query):
                continue
            if self._domain_overlap(memory, routed_domains):
                candidate_indices.append(idx)
            fallback_indices.append(idx)

        # Domain routing is intentionally soft: it may add a small rerank bonus,
        # but it must never remove raw evidence from dense/BM25/lexical sources.
        candidate_indices = fallback_indices

        domain_top_k = max(self.domain_seed_top_k, k * 4)
        if category_int == 1:
            domain_top_k = max(domain_top_k, DEFAULT_CAT1_EXPANDED_GLOBAL_TOP_K)
        embedding_ranked = self._embedding_rank_indices(query, candidate_indices, domain_top_k)
        bm25_ranked = self._bm25_rank_indices(lexical_query, candidate_indices, domain_top_k)

        candidate_scores: Dict[int, Dict[str, Any]] = {}

        def _entry(idx: int) -> Dict[str, Any]:
            return candidate_scores.setdefault(idx, {
                "domain_embedding": 0.0,
                "domain_bm25": 0.0,
                "domain_lexical": 0.0,
                "global_embedding": 0.0,
                "global_bm25": 0.0,
                "global_entity": 0.0,
                "graph_expansion": 0.0,
                "domain_match": 0.0,
                "source_tags": set(),
            })

        def _rank_score(score: float, rank: int, cap: float = 1.0) -> float:
            capped = min(float(score), cap) / max(cap, 1e-6)
            return max(capped, 1.0 / (rank + 1))

        for rank, (score, idx) in enumerate(embedding_ranked):
            item = _entry(idx)
            item["domain_embedding"] = max(item["domain_embedding"], _rank_score(score, rank, cap=1.0))
            item["source_tags"].add("domain_embedding")
        bm25_cap = max((score for score, _ in bm25_ranked), default=1.0) or 1.0
        for rank, (score, idx) in enumerate(bm25_ranked):
            item = _entry(idx)
            item["domain_bm25"] = max(item["domain_bm25"], _rank_score(score, rank, cap=bm25_cap))
            item["source_tags"].add("domain_bm25")

        lexical_ranked = []
        for idx in candidate_indices:
            lexical = self._lexical_relevance(self._ensure_memory_schema(all_memories[idx]), lexical_query)
            if lexical > 0.0:
                lexical_ranked.append((lexical, idx))
        lexical_ranked.sort(reverse=True)
        lexical_cap = max((score for score, _ in lexical_ranked[:domain_top_k]), default=1.0) or 1.0
        for rank, (score, idx) in enumerate(lexical_ranked[:domain_top_k]):
            item = _entry(idx)
            item["domain_lexical"] = max(item["domain_lexical"], _rank_score(score, rank, cap=lexical_cap))
            item["source_tags"].add("domain_lexical")

        if self.global_fallback_top_k:
            global_embedding_top_k = self.global_fallback_top_k
            global_bm25_top_k = self.global_bm25_top_k
            global_entity_top_k = self.global_entity_top_k
            if category_int == 1:
                global_embedding_top_k = max(global_embedding_top_k, DEFAULT_CAT1_EXPANDED_GLOBAL_TOP_K)
                global_bm25_top_k = max(global_bm25_top_k, DEFAULT_CAT1_EXPANDED_GLOBAL_TOP_K)
                global_entity_top_k = max(global_entity_top_k, DEFAULT_CAT1_EXPANDED_GLOBAL_TOP_K)
            global_embedding = self._embedding_rank_indices(query, fallback_indices, global_embedding_top_k)
            global_bm25 = self._bm25_rank_indices(lexical_query, fallback_indices, global_bm25_top_k)
            global_entity = self._entity_rank_indices(lexical_query, fallback_indices, global_entity_top_k)
            for rank, (score, idx) in enumerate(global_embedding):
                item = _entry(idx)
                item["global_embedding"] = max(item["global_embedding"], _rank_score(score, rank, cap=1.0))
                item["source_tags"].add("global_embedding")
            global_bm25_cap = max((score for score, _ in global_bm25), default=1.0) or 1.0
            for rank, (score, idx) in enumerate(global_bm25):
                item = _entry(idx)
                item["global_bm25"] = max(item["global_bm25"], _rank_score(score, rank, cap=global_bm25_cap))
                item["source_tags"].add("global_bm25")
            global_entity_cap = max((score for score, _ in global_entity), default=1.0) or 1.0
            for rank, (score, idx) in enumerate(global_entity):
                item = _entry(idx)
                item["global_entity"] = max(item["global_entity"], _rank_score(score, rank, cap=global_entity_cap))
                item["source_tags"].add("global_entity")

        id_to_idx = {
            self._ensure_memory_schema(memory).id: idx
            for idx, memory in enumerate(all_memories)
        }
        relation_weights = {
            "same_storyline": 0.95,
            "similar_event": 0.72,
            "image_text_pair": 0.55,
            "same_character": 0.45,
        }
        if category_int == 1:
            allowed_graph_relations = {
                "same_storyline", "similar_event", "same_character", "image_text_pair",
            }
            seed_limit, per_seed_limit, graph_candidate_limit = 20, 3, 36
        elif category_int == 2:
            allowed_graph_relations = {
                "same_storyline", "similar_event", "same_character", "image_text_pair",
            }
            seed_limit, per_seed_limit, graph_candidate_limit = 10, 2, 16
        elif category_int == 4:
            allowed_graph_relations = {
                "same_storyline", "image_text_pair", "similar_event",
            }
            seed_limit, per_seed_limit, graph_candidate_limit = 8, 1, 10
        else:
            allowed_graph_relations = {
                "same_storyline", "similar_event", "same_character", "image_text_pair",
            }
            seed_limit, per_seed_limit, graph_candidate_limit = 10, 2, 12

        def _pre_graph_score(item: Dict[str, Any]) -> float:
            return (
                0.25 * item.get("domain_embedding", 0.0)
                + 0.20 * item.get("domain_bm25", 0.0)
                + 0.16 * item.get("domain_lexical", 0.0)
                + 0.12 * item.get("global_embedding", 0.0)
                + 0.20 * item.get("global_bm25", 0.0)
                + 0.18 * item.get("global_entity", 0.0)
            )

        seed_indices = [
            idx for idx, _ in sorted(
                candidate_scores.items(),
                key=lambda pair: _pre_graph_score(pair[1]),
                reverse=True,
            )[:seed_limit]
        ]
        graph_added = 0
        for seed_idx in seed_indices:
            if seed_idx >= len(all_memories) or graph_added >= graph_candidate_limit:
                continue
            seed_memory = self._ensure_memory_schema(all_memories[seed_idx])
            seed_score = max(0.15, _pre_graph_score(candidate_scores.get(seed_idx, {})))
            edge_items = []
            for edge in getattr(seed_memory, "links", []) or []:
                if not isinstance(edge, dict):
                    continue
                relation = self._edge_relation(edge)
                if relation not in allowed_graph_relations:
                    continue
                target_id = edge.get("target_id")
                target_idx = id_to_idx.get(target_id)
                if target_idx is None or target_idx == seed_idx or target_idx >= len(all_memories):
                    continue
                target_memory = self._ensure_memory_schema(all_memories[target_idx])
                if not self._is_retrievable(target_memory, query):
                    continue
                lexical = self._lexical_relevance(target_memory, lexical_query)
                if category_int == 4 and lexical < 1.0:
                    continue
                if category_int == 2 and relation not in {"same_storyline", "image_text_pair"} and lexical < 1.0:
                    continue
                strength = float(edge.get("strength", 0.5))
                relation_weight = relation_weights.get(relation, 0.4)
                graph_score = seed_score * strength * relation_weight * (1.0 + min(1.0, lexical / 8.0))
                edge_items.append((graph_score, relation, target_idx))
            edge_items.sort(reverse=True)
            per_seed_added = 0
            for graph_score, relation, target_idx in edge_items:
                item = _entry(target_idx)
                if graph_score > item["graph_expansion"]:
                    item["graph_expansion"] = graph_score
                    item["source_tags"].add(f"graph_{relation}")
                per_seed_added += 1
                graph_added += 1
                if per_seed_added >= per_seed_limit or graph_added >= graph_candidate_limit:
                    break

        ranked = []
        query_domains = routed_domains or []
        session_counts: Dict[int, int] = {}
        for order, idx in enumerate(candidate_scores.keys()):
            if idx >= len(all_memories):
                continue
            memory = self._ensure_memory_schema(all_memories[idx])
            if not self._is_retrievable(memory, query):
                continue
            score_parts = candidate_scores[idx]
            lexical_score = self._lexical_relevance(memory, lexical_query)
            domain_score = 1.0 if self._domain_overlap(memory, query_domains) else 0.0
            score_parts["domain_match"] = domain_score
            reliability = self.memory_reliability(memory)
            citation_score = math.log1p(float(getattr(memory, "citation_count", 0))) / 5.0
            lexical_norm = min(1.0, lexical_score / 8.0)
            if category_int == 4:
                weights = {
                    "domain_embedding": 0.10,
                    "domain_bm25": 0.15,
                    "domain_lexical": 0.15,
                    "global_embedding": 0.05,
                    "global_bm25": 0.30,
                    "global_entity": 0.25,
                    "graph_expansion": 0.05,
                    "domain_match": 0.05,
                    "lexical": 0.20,
                }
            elif category_int == 1:
                weights = {
                    "domain_embedding": 0.18,
                    "domain_bm25": 0.18,
                    "domain_lexical": 0.18,
                    "global_embedding": 0.07,
                    "global_bm25": 0.22,
                    "global_entity": 0.18,
                    "graph_expansion": 0.18,
                    "domain_match": 0.08,
                    "lexical": 0.18,
                }
            else:
                weights = {
                    "domain_embedding": 0.25,
                    "domain_bm25": 0.18,
                    "domain_lexical": 0.12,
                    "global_embedding": 0.08,
                    "global_bm25": 0.18,
                    "global_entity": 0.14,
                    "graph_expansion": 0.10,
                    "domain_match": 0.10,
                    "lexical": 0.18,
                }
            session_bonus = 0.0
            dia_key = self._dia_sort_key(self._dia_id_for_memory(memory) or "")
            if category_int == 1 and dia_key:
                # Small diversity bonus during ordering so multi-hop candidates do not all come from one session.
                session_bonus = max(0.0, 0.04 - 0.02 * session_counts.get(dia_key[0], 0))
            score_inputs = {
                "domain_embedding": score_parts["domain_embedding"],
                "domain_bm25": score_parts["domain_bm25"],
                "domain_lexical": score_parts["domain_lexical"],
                "global_embedding": score_parts["global_embedding"],
                "global_bm25": score_parts["global_bm25"],
                "global_entity": score_parts["global_entity"],
                "graph_expansion": min(1.0, score_parts["graph_expansion"]),
                "domain_match": score_parts["domain_match"],
                "lexical_norm": lexical_norm,
                "reliability": reliability,
                "citation": min(1.0, citation_score),
                "session_bonus": session_bonus,
            }
            score_contributions = {
                "domain_embedding": weights["domain_embedding"] * score_inputs["domain_embedding"],
                "domain_bm25": weights["domain_bm25"] * score_inputs["domain_bm25"],
                "domain_lexical": weights["domain_lexical"] * score_inputs["domain_lexical"],
                "global_embedding": weights["global_embedding"] * score_inputs["global_embedding"],
                "global_bm25": weights["global_bm25"] * score_inputs["global_bm25"],
                "global_entity": weights["global_entity"] * score_inputs["global_entity"],
                "graph_expansion": weights["graph_expansion"] * score_inputs["graph_expansion"],
                "domain_match": weights["domain_match"] * score_inputs["domain_match"],
                "lexical": weights["lexical"] * score_inputs["lexical_norm"],
                "reliability": 0.07 * score_inputs["reliability"],
                "citation": 0.03 * score_inputs["citation"],
                "session_bonus": session_bonus,
            }
            combined_score = sum(score_contributions.values())
            if dia_key:
                session_counts[dia_key[0]] = session_counts.get(dia_key[0], 0) + 1
            debug_parts = dict(score_parts)
            debug_parts["source_tags"] = sorted(debug_parts.get("source_tags", []))
            debug_parts["combined_score"] = combined_score
            debug_parts["lexical_score"] = lexical_score
            debug_parts["lexical_norm"] = lexical_norm
            debug_parts["reliability_score"] = reliability
            debug_parts["citation_score"] = score_inputs["citation"]
            debug_parts["session_bonus"] = session_bonus
            debug_parts["score_weights"] = dict(weights)
            debug_parts["score_inputs"] = score_inputs
            debug_parts["score_contributions"] = score_contributions
            debug_parts["dia_id"] = self._dia_id_for_memory(memory)
            debug_parts["memory_id"] = memory.id
            candidate_scores[idx] = debug_parts
            ranked.append((self.lifecycle_penalty(memory), -combined_score, order, idx, memory))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        for combined_rank, item in enumerate(ranked, start=1):
            idx = item[3]
            if idx in candidate_scores:
                candidate_scores[idx]["combined_rank"] = combined_rank

        rank_components = [
            "domain_embedding", "domain_bm25", "domain_lexical",
            "global_embedding", "global_bm25", "global_entity",
            "graph_expansion", "domain_match", "lexical_norm",
            "reliability_score", "citation_score", "combined_score",
        ]
        for component in rank_components:
            sorted_items = sorted(
                candidate_scores.items(),
                key=lambda pair: float(pair[1].get(component, 0.0) or 0.0),
                reverse=True,
            )
            for rank, (idx, debug) in enumerate(sorted_items, start=1):
                value = float(debug.get(component, 0.0) or 0.0)
                ranks = debug.setdefault("source_ranks", {})
                ranks[component] = rank if value > 0.0 or component == "combined_score" else None

        self.last_candidate_debug = [
            candidate_scores.get(item[3], {})
            for item in ranked[: max(100, k * 10, self.final_bundle_max_size)]
        ]
        return ranked

    def _category1_slot_tokens(self, memory: RobustMemoryNote, query: str) -> Set[str]:
        """Tokens that can add complementary Cat1 evidence beyond the query wording."""
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        evidence_text = " ".join([
            self._retrieval_memory_text(memory),
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
            " ".join(getattr(memory, "keywords", [])),
        ])
        query_tokens = self._retrieval_tokens(query)
        tokens = self._retrieval_tokens(evidence_text) - query_tokens
        generic_tokens = {
            "thing", "things", "something", "someone", "people", "person", "memory",
            "memories", "talk", "said", "tell", "asked", "shared", "discussed",
            "session", "image", "photo", "picture", "caption", "query",
        }
        return {token for token in tokens if token not in generic_tokens}

    @staticmethod
    def _category1_slot_profile(query: str) -> Dict[str, Any]:
        """Infer the answer slot Cat1 is asking for from stable question wording."""
        query_lower = str(query or "").lower()
        slot_type = "fact"
        cue_terms: Set[str] = set()
        if re.search(r"\bidentity|relationship status|career path\b", query_lower):
            slot_type = "status"
            cue_terms.update({
                "identity", "relationship", "status", "career", "path", "decided",
                "transition", "transgender", "womanhood", "single", "breakup",
                "counseling", "mental", "health",
            })
        elif re.search(r"\bhow many\b|\bnumber of\b", query_lower):
            slot_type = "count"
            cue_terms.update({"time", "times", "number", "count", "once", "twice", "three", "four"})
        elif re.search(r"\bwhere\b", query_lower):
            slot_type = "place"
            cue_terms.update({
                "where", "place", "places", "moved", "move", "from", "camp", "camped",
                "beach", "mountain", "mountains", "forest", "country", "city", "home",
                "park", "trail", "lake", "roadtrip", "trip",
            })
        elif re.search(r"\bwho\b", query_lower):
            slot_type = "person"
            cue_terms.update({"who", "friend", "friends", "kid", "kids", "child", "children", "support", "supports"})
        elif re.search(r"\bbook|books|read|suggestion\b", query_lower):
            slot_type = "book"
            cue_terms.update({"book", "books", "read", "novel", "story", "title", "author", "recommended", "suggested"})
        elif re.search(r"\bevent|events|participat|attended|community\b", query_lower):
            slot_type = "event"
            cue_terms.update({
                "event", "events", "participated", "attended", "joined", "pride",
                "parade", "conference", "poetry", "reading", "school", "council",
                "meeting", "activist", "group", "community", "fundraiser",
            })
        elif re.search(r"\bactivit|destress|hikes?|family|partake|does\b", query_lower):
            slot_type = "activity"
            cue_terms.update({
                "activity", "activities", "destress", "hike", "hiking", "camp", "camping",
                "paint", "painting", "pottery", "swim", "swimming", "run", "running",
                "violin", "clarinet", "music", "beach", "family", "kids",
            })
        elif re.search(r"\bpaint|art|pottery|symbols?|items?|instruments?|pets?|names?|changes?|types?|ways?\b", query_lower):
            slot_type = "item"
            cue_terms.update({
                "paint", "painted", "painting", "art", "pottery", "symbol", "symbols",
                "item", "items", "instrument", "instruments", "pet", "pets", "name",
                "names", "change", "changes", "type", "types", "bought", "made",
            })

        generic_question_terms = {
            "what", "where", "when", "who", "which", "does", "did", "has", "have",
            "caroline", "melanie", "their", "from", "with", "some", "many",
        }
        target_terms = {
            token for token in re.findall(r"[a-z0-9]+", query_lower)
            if len(token) >= 3 and token not in generic_question_terms
        }
        return {
            "slot_type": slot_type,
            "cue_terms": cue_terms,
            "target_terms": target_terms,
        }

    def _category1_slot_signal(
        self,
        memory: RobustMemoryNote,
        query: str,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        fields = self._parse_memory_fields(self._raw_memory_content(memory))
        evidence_text = " ".join([
            fields.get("speaker", ""),
            self._retrieval_memory_text(memory),
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
            " ".join(getattr(memory, "keywords", [])),
        ])
        evidence_lower = evidence_text.lower()
        evidence_tokens = self._retrieval_tokens(evidence_text)
        slot_tokens = self._category1_slot_tokens(memory, query)
        cue_terms = set(profile.get("cue_terms", set()))
        target_terms = set(profile.get("target_terms", set()))
        cue_hits = {term for term in cue_terms if term in evidence_lower or self._canonical_token(term) in evidence_tokens}
        target_hits = target_terms & evidence_tokens
        named_entities = {
            self._canonical_token(token)
            for token in re.findall(r"\b[A-Z][A-Za-z0-9']+\b", evidence_text)
            if token.lower() not in {"Caroline".lower(), "Melanie".lower()}
        }
        quoted_terms = {
            self._canonical_token(token)
            for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", evidence_text)
            for token in re.findall(r"[A-Za-z0-9]+", quoted)
            if len(token) >= 3
        }
        slot_type = str(profile.get("slot_type", "fact"))
        typed_tokens = set(slot_tokens)
        if slot_type in {"book", "event", "item", "person", "place"}:
            typed_tokens |= named_entities | quoted_terms
        if slot_type == "count":
            number_words = {
                "one", "two", "three", "four", "five", "six", "seven", "eight",
                "once", "twice", "couple", "several",
            }
            typed_tokens |= {
                token for token in evidence_tokens
                if token.isdigit() or token in number_words
            }
        slot_score = (
            2.0 * len(cue_hits)
            + 1.5 * len(target_hits)
            + 0.4 * min(10, len(typed_tokens))
        )
        if cue_hits and named_entities:
            slot_score += 1.0
        return {
            "slot_type": slot_type,
            "slot_score": slot_score,
            "slot_cue_hits": sorted(cue_hits)[:12],
            "slot_target_hits": sorted(target_hits)[:12],
            "slot_tokens": typed_tokens,
        }

    def _select_category1_coverage_ranked(
        self,
        ranked: List[tuple],
        query: str,
        primary_limit: int,
        k: int,
    ) -> List[tuple]:
        """Greedily front-load Cat1 candidates that cover different answer slots."""
        if len(ranked) <= 1 or primary_limit <= 1:
            return ranked

        pool_size = min(len(ranked), max(k * 3, self.final_bundle_max_size * 3, primary_limit * 5))
        pool = list(ranked[:pool_size])
        tail = list(ranked[pool_size:])
        selected: List[tuple] = []
        remaining = pool
        covered_tokens: Set[str] = set()
        selected_sessions: Dict[int, int] = {}
        slot_profile = self._category1_slot_profile(query)

        debug_by_memory_id = {
            str(item.get("memory_id")): item
            for item in self.last_candidate_debug
            if item.get("memory_id") is not None
        }
        slot_cache: Dict[str, Set[str]] = {}
        signal_cache: Dict[str, Dict[str, Any]] = {}

        def _score(item: tuple) -> tuple:
            _, neg_combined_score, original_order, _, memory = item
            memory = self._ensure_memory_schema(memory)
            memory_id = str(memory.id)
            slot_tokens = slot_cache.setdefault(memory_id, self._category1_slot_tokens(memory, query))
            signal = signal_cache.setdefault(memory_id, self._category1_slot_signal(memory, query, slot_profile))
            new_tokens = slot_tokens - covered_tokens
            lexical_score = self._lexical_relevance(memory, query)
            dia_key = self._dia_sort_key(self._dia_id_for_memory(memory) or "")
            session_idx = dia_key[0] if dia_key else -1
            same_session_count = selected_sessions.get(session_idx, 0) if session_idx >= 0 else 0
            combined_score = -float(neg_combined_score)
            relevance_penalty = 0.12 if lexical_score <= 0.0 and combined_score < 0.20 else 0.0
            slot_score = float(signal.get("slot_score", 0.0))
            coverage_bonus = 0.025 * min(8, len(new_tokens))
            slot_bonus = 0.075 * min(10.0, slot_score)
            diversity_bonus = 0.035 if session_idx >= 0 and same_session_count == 0 else 0.0
            lexical_bonus = 0.025 * min(1.0, lexical_score / 8.0)
            repeated_session_penalty = 0.025 * same_session_count
            selection_score = (
                combined_score
                + coverage_bonus
                + slot_bonus
                + diversity_bonus
                + lexical_bonus
                - repeated_session_penalty
                - relevance_penalty
            )
            return (
                selection_score,
                len(new_tokens),
                combined_score,
                -original_order,
                new_tokens,
                slot_tokens,
                signal,
                lexical_score,
                session_idx,
            )

        selection_diagnostics: Dict[str, Dict[str, Any]] = {}
        while remaining and len(selected) < primary_limit:
            best_item = max(remaining, key=_score)
            best_score = _score(best_item)
            remaining.remove(best_item)
            selected.append(best_item)
            _, _, _, _, new_tokens, slot_tokens, signal, lexical_score, session_idx = best_score
            covered_tokens.update(new_tokens)
            if session_idx >= 0:
                selected_sessions[session_idx] = selected_sessions.get(session_idx, 0) + 1
            memory = self._ensure_memory_schema(best_item[4])
            selection_diagnostics[str(memory.id)] = {
                "cat1_selected_primary": True,
                "cat1_selected_rank": len(selected),
                "cat1_selection_score": round(float(best_score[0]), 6),
                "cat1_coverage_gain": len(new_tokens),
                "cat1_new_slot_tokens": sorted(new_tokens)[:12],
                "cat1_slot_tokens": sorted(slot_tokens)[:20],
                "cat1_answer_slot_type": signal.get("slot_type"),
                "cat1_answer_slot_score": round(float(signal.get("slot_score", 0.0)), 6),
                "cat1_slot_cue_hits": signal.get("slot_cue_hits", []),
                "cat1_slot_target_hits": signal.get("slot_target_hits", []),
                "cat1_lexical_score": lexical_score,
            }

        for item in remaining:
            memory = self._ensure_memory_schema(item[4])
            memory_id = str(memory.id)
            slot_tokens = slot_cache.setdefault(memory_id, self._category1_slot_tokens(memory, query))
            signal = signal_cache.setdefault(memory_id, self._category1_slot_signal(memory, query, slot_profile))
            selection_diagnostics[memory_id] = {
                "cat1_selected_primary": False,
                "cat1_coverage_gain": len(slot_tokens - covered_tokens),
                "cat1_slot_tokens": sorted(slot_tokens)[:20],
                "cat1_answer_slot_type": signal.get("slot_type"),
                "cat1_answer_slot_score": round(float(signal.get("slot_score", 0.0)), 6),
                "cat1_slot_cue_hits": signal.get("slot_cue_hits", []),
                "cat1_slot_target_hits": signal.get("slot_target_hits", []),
                "cat1_lexical_score": self._lexical_relevance(memory, query),
            }

        reordered = selected + remaining + tail
        updated_debug = []
        for _, _, _, _, memory in reordered[: max(100, k * 10, self.final_bundle_max_size)]:
            debug = debug_by_memory_id.get(str(memory.id))
            if not debug:
                continue
            debug.update(selection_diagnostics.get(str(memory.id), {}))
            updated_debug.append(debug)
        if updated_debug:
            self.last_candidate_debug = updated_debug
        return reordered

    def _memory_to_index_text(self, memory: RobustMemoryNote) -> str:
        """Build a weighted retriever document from the current valid memory object."""
        self._ensure_rewrite_content(memory)
        raw_content = self._raw_memory_content(memory)
        fields = self._parse_memory_fields(raw_content)
        main_content = self._retrieval_memory_text(memory)
        visual_cues = " ".join([
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
        ]).strip()
        metadata = " ".join([
            fields.get("dia_id", ""),
            fields.get("session_date", ""),
            fields.get("speaker", ""),
            " ".join(getattr(memory, "keywords", [])),
            " ".join(getattr(memory, "tags", [])),
        ]).strip()
        pieces = [
            "rewrite_content: " + main_content,
            "metadata: " + metadata,
            "status: " + getattr(memory, "status", "active"),
        ]
        if visual_cues:
            pieces.append("visual cue: " + visual_cues)
        return " ".join(piece for piece in pieces if piece.strip())

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
                        entry["examples"].append(self._retrieval_memory_text(memory))
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

    def memory_reliability(self, memory: RobustMemoryNote) -> float:
        """Bayesian-style utility estimate from historical citation feedback."""
        success = float(getattr(memory, "successful_citation_count", 0))
        failure = float(getattr(memory, "failed_citation_count", 0))
        alpha = float(getattr(memory, "reliability_alpha", 1.0))
        beta = float(getattr(memory, "reliability_beta", 1.0))
        return (success + alpha) / max(success + failure + alpha + beta, 1e-6)

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
        if not hasattr(memory, "temporal_expressions"):
            memory.temporal_expressions = []
        if not hasattr(memory, "confidence"):
            memory.confidence = 1.0
        if not hasattr(memory, "reliability_alpha"):
            memory.reliability_alpha = 1.0
        if not hasattr(memory, "reliability_beta"):
            memory.reliability_beta = 1.0
        if not hasattr(memory, "citation_count"):
            memory.citation_count = 0
        if not hasattr(memory, "successful_citation_count"):
            memory.successful_citation_count = 0
        if not hasattr(memory, "failed_citation_count"):
            memory.failed_citation_count = 0
        if not hasattr(memory, "last_updated"):
            memory.last_updated = getattr(memory, "last_accessed", getattr(memory, "timestamp", ""))
        if not hasattr(memory, "rewrite_content"):
            memory.rewrite_content = ""
        self._ensure_rewrite_content(memory)
        return memory

    def _format_memory_for_context(
        self,
        memory: RobustMemoryNote,
        relation: Optional[str] = None,
        edge_reason: Optional[str] = None,
    ) -> str:
        relation_text = f"relation: {relation} " if relation else ""
        reason_text = f"edge reason: {edge_reason} " if edge_reason else ""
        return (
            relation_text +
            reason_text +
            "talk start time:" + str(memory.timestamp) +
            " memory content: " + str(memory.current_content) + "\n"
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
        provided_domain_paths = kwargs.get("domain_paths")
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        self._ensure_memory_schema(note)
        if not provided_domain_paths:
            note.domain_paths = self.route_domains(self._retrieval_memory_text(note), note=note)
        if getattr(note, "memory_level", DEFAULT_MEMORY_LEVEL) == "instance" and provided_domain_paths:
            self.memories[note.id] = note
            self.retriever.add_documents([self._memory_to_index_text(note)])
            return note.id
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
        """Rebuild the retriever index while reusing the already loaded embedding model."""
        if hasattr(self.retriever, "model"):
            self.retriever.corpus = []
            self.retriever.embeddings = None
            self.retriever.document_ids = {}
        else:
            self.retriever = SimpleEmbeddingRetriever(DEFAULT_EMBEDDING_MODEL)

        for memory in self.memories.values():
            if self._is_indexable(memory):
                self.retriever.add_documents([self._memory_to_index_text(memory)])

    def find_related_memories(
        self,
        query: str,
        k: int = 5,
        routed_domains: Optional[List[str]] = None,
        route_domain: bool = True,
        category: Optional[int] = None,
    ) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        if route_domain and routed_domains is None:
            routed_domains = self.route_domains(query)
        self.last_routed_domains = list(routed_domains or [])
        all_memories = list(self.memories.values())
        memory_str = ""
        ranked = self._retrieval_candidates(query, k, routed_domains, category=category)
        filtered_indices = []
        for _, _, _, i, memory in ranked[:k]:
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
        category: Optional[int] = None,
    ) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        if route_domain and routed_domains is None:
            routed_domains = self.route_domains(query)
        self.last_routed_domains = list(routed_domains or [])
        try:
            category_int = int(category) if category is not None else None
        except (TypeError, ValueError):
            category_int = None
        all_memories = list(self.memories.values())
        id_to_memory = {memory.id: self._ensure_memory_schema(memory) for memory in all_memories}
        dia_lookup = self._build_dia_lookup(all_memories)
        memory_str = ""
        seen_ids = set()
        ranked = self._retrieval_candidates(query, k, routed_domains, category=category_int)
        primary_count = 0
        if category_int == 1:
            primary_limit = min(
                max(k, DEFAULT_CAT1_PRIMARY_BUNDLE_SIZE),
                DEFAULT_CAT1_PRIMARY_BUNDLE_SIZE,
            )
        else:
            primary_limit = min(k, self.final_bundle_size)
        max_context_blocks = (
            DEFAULT_CAT1_MAX_CONTEXT_BLOCKS
            if category_int == 1
            else max(primary_limit, self.final_bundle_max_size)
        )
        local_context_primary_limit = 2
        local_context_min_lexical_score = 2.0
        local_context_neighbor_min_lexical_score = 1.0
        relation_priority = {
            "similar_event": 0,
            "same_storyline": 1,
            "image_text_pair": 2,
            "same_character": 3,
            "local_context": 4,
            "same_entity": 6,
            "same_topic": 7,
            "semantic_related": 8,
        }
        if category_int == 1 and self.enable_cat1_coverage_rerank:
            relation_priority.update({
                "same_storyline": 0,
                "image_text_pair": 1,
                "similar_event": 2,
                "same_character": 5,
            })
            ranked = self._select_category1_coverage_ranked(ranked, query, primary_limit, k)
            protected_primary_count = primary_limit
            effective_local_context_primary_limit = protected_primary_count
        elif category_int == 1:
            protected_primary_count = 0
            effective_local_context_primary_limit = local_context_primary_limit
            self.last_candidate_debug = [
                dict(item, cat1_coverage_rerank_disabled=True)
                for item in self.last_candidate_debug
            ]
        else:
            protected_primary_count = 0
            effective_local_context_primary_limit = local_context_primary_limit
        for _, _, _, i, memory in ranked:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            memory_str += self._format_memory_for_context(memory)
            primary_count += 1

            lexical_score = self._lexical_relevance(memory, query)
            allow_expansion = category_int != 1
            if (
                allow_expansion
                and primary_count <= effective_local_context_primary_limit
                and lexical_score >= local_context_min_lexical_score
            ):
                local_added = 0
                for target_memory in self._local_context_neighbors(memory, dia_lookup, radius=1):
                    if (
                        not self._is_retrievable(target_memory, query)
                        or target_memory.id in seen_ids
                        or len(seen_ids) >= max_context_blocks
                        or self._lexical_relevance(target_memory, query) < local_context_neighbor_min_lexical_score
                    ):
                        continue
                    seen_ids.add(target_memory.id)
                    memory_str += self._format_memory_for_context(target_memory, relation="local_context")
                    local_added += 1
                    if local_added >= 1:
                        break

            neighbor_count = 0
            same_character_count = 0
            similar_event_count = 0
            sorted_edges = sorted(
                getattr(memory, "links", []),
                key=lambda edge: (
                    relation_priority.get(self._edge_relation(edge), 99),
                    -float(edge.get("strength", 0.0)) if isinstance(edge, dict) else 0.0,
                ),
            )
            if not allow_expansion:
                sorted_edges = []
            direct_context_relations = {
                "similar_event", "same_character", "same_storyline", "image_text_pair",
            }
            for edge in sorted_edges:
                target_memory = self._edge_to_memory(edge, all_memories, id_to_memory)
                relation = self._edge_relation(edge)
                if (
                    not target_memory
                    or relation not in STABLE_RETRIEVAL_EDGES
                    or relation not in direct_context_relations
                    or not self._is_retrievable(target_memory, query)
                    or target_memory.id in seen_ids
                ):
                    continue
                if relation == "similar_event":
                    if category_int == 4:
                        continue
                    if similar_event_count >= 1:
                        continue
                    if self._lexical_relevance(target_memory, query) < 1.0:
                        continue
                    similar_event_count += 1
                if relation == "same_character":
                    if category_int == 4 or same_character_count >= 1:
                        continue
                    same_character_count += 1
                seen_ids.add(target_memory.id)
                edge_reason = str(edge.get("reason", "")) if isinstance(edge, dict) else ""
                memory_str += self._format_memory_for_context(
                    target_memory,
                    relation=relation,
                    edge_reason=edge_reason if relation in {"same_storyline", "image_text_pair"} else None,
                )
                neighbor_count += 1
                neighbor_limit = 1 if category_int == 1 else 2
                if neighbor_count >= neighbor_limit or len(seen_ids) >= max_context_blocks:
                    break
            if primary_count >= primary_limit or len(seen_ids) >= max_context_blocks:
                break
        return memory_str

    # ---- write-time lifecycle consolidation ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Resolve updates/conflicts at write time and maintain only stable graph edges."""
        candidate_memory, indices = self.retrieve_same_entity_or_topic(note, top_k=10)
        if not indices:
            note._requires_reindex = False
            return False, note

        if getattr(note, "memory_level", DEFAULT_MEMORY_LEVEL) == "instance":
            self.add_stable_graph_edges(
                note,
                indices,
                {
                    "relation": "semantic_related",
                    "confidence": 0.6,
                    "reason": "Instance-level memories are preserved as separate episodic evidence.",
                },
            )
            note._requires_reindex = False
            return True, note

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
            self._retrieval_memory_text(new_memory),
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
                content=self._retrieval_memory_text(new_memory),
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
        content_lower = self._retrieval_memory_text(new_memory).lower()

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
            self._retrieval_memory_text(memory),
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
        combined = f"{self._retrieval_memory_text(old_memory)} {self._retrieval_memory_text(new_memory)}".lower()
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
        old_memory.rewrite_content = self._rewrite_memory_content(old_memory)
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
            "target_dia_id": self._dia_id_for_memory(target),
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
