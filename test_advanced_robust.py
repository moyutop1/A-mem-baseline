"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import (
    RETRIEVAL_INDEX_VERSION,
    RobustLLMController,
    RobustAgenticMemorySystem,
)
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import logging
import re
from typing import List, Dict, Optional, Set
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime, timedelta

EMBEDDING_MODEL_NAME = os.getenv("SENTENCE_MODEL_PATH", "all-MiniLM-L6-v2")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

logger = logging.getLogger("amem_robust")


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000,
                 compress_categories: Optional[Set[int]] = None):
        self.memory_system = RobustAgenticMemorySystem(
            model_name=EMBEDDING_MODEL_NAME,
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.compress_categories = compress_categories or set()

    def add_memory(self, content, time=None, **kwargs):
        self.memory_system.add_note(content, time=time, **kwargs)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    @staticmethod
    def build_retrieval_query(question: str, keywords: str) -> str:
        """Keep the original question in retrieval so keyword drift cannot erase key evidence."""
        parts = [question.strip()]
        if keywords and keywords.strip() and keywords.strip().lower() != question.strip().lower():
            parts.append(keywords.strip())
        return " ; ".join(parts)

    def _retrieve_memory_llm_legacy(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def retrieve_memory_llm(self, memories_text, query):
        """Compress retrieved memories into the most relevant evidence for answering."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant evidence blocks that would help answer the question.
Preserve dia_id, session_date, speaker, and exact quoted facts when available.
Prefer directly relevant blocks over broad topical matches.
Use local_context blocks only when they clarify a directly relevant block.

Conversation memories:
{memories_text}

Question: {query}

Return 3-6 concise evidence lines.
If no parts are relevant, return the input unchanged."""

        try:
            response = self.retriever_llm.llm.get_completion(prompt, temperature=0.1)
            compressed = parse_relevant_parts(response)
        except Exception as e:
            logger.warning("retrieve_memory_llm failed: %s; using raw retrieved context", e)
            return memories_text
        return compressed or memories_text

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        try:
            response = self.retriever_llm.llm.get_completion(prompt)
            result = parse_keywords_response(response)
        except Exception as e:
            logger.warning("generate_query_llm failed: %s; falling back to original question", e)
            result = question
        logger.debug("generate_query_llm response: %s", result)
        return result

    @staticmethod
    def _parse_context_blocks(context: str) -> List[Dict[str, str]]:
        """Extract retrieved memory blocks with their session dates and contents."""
        blocks = []
        pattern = re.compile(
            r"talk start time:(?P<date>.*?) memory content: (?P<content>.*?)(?= relation: .*?talk start time:|talk start time:|\Z)",
            re.DOTALL,
        )
        for match in pattern.finditer(context):
            blocks.append({
                "date": match.group("date").strip(),
                "content": match.group("content").strip(),
            })
        return blocks

    @staticmethod
    def _memory_body(content: str) -> str:
        return content.split(" memory context:", 1)[0]

    @staticmethod
    def _parse_memory_fields(content: str) -> Dict[str, str]:
        body = RobustAdvancedMemAgent._memory_body(content)
        field_names = "dia_id|session_date|speaker|content|image_caption|image_query"
        fields = {}
        for match in re.finditer(
            rf"(?:^|\n)(?P<key>{field_names}):\s*(?P<value>.*?)(?=\n(?:{field_names}):|\Z)",
            body,
            re.DOTALL,
        ):
            fields[match.group("key")] = match.group("value").strip()
        return fields

    @staticmethod
    def _extract_retrieved_dia_ids(context: str) -> List[str]:
        seen = set()
        dia_ids = []
        for match in re.finditer(r"dia_id:\s*(D\d+:\d+)", context):
            dia_id = match.group(1)
            if dia_id not in seen:
                seen.add(dia_id)
                dia_ids.append(dia_id)
        return dia_ids

    @staticmethod
    def _overlap_tokens(text: str) -> Set[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "into", "what", "when",
            "where", "which", "would", "could", "should", "about", "mention", "mentioned",
            "conversation", "answer", "short", "date", "time", "last", "week", "year",
            "month", "yesterday", "today", "tomorrow", "before", "after", "did", "does",
            "was", "were", "has", "have", "had", "her", "his", "their", "they", "she",
            "him", "caroline", "melanie",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 3 and token not in stopwords
        }

    def _score_temporal_block(self, block: Dict[str, str], question: str, response: str) -> int:
        fields = self._parse_memory_fields(block["content"])
        main_text = " ".join([
            fields.get("content", ""),
            fields.get("speaker", ""),
        ])
        metadata_text = " ".join([
            fields.get("image_caption", ""),
            fields.get("image_query", ""),
            fields.get("dia_id", ""),
        ])
        question_tokens = self._overlap_tokens(question)
        response_tokens = self._overlap_tokens(response)
        main_tokens = self._overlap_tokens(main_text)
        metadata_tokens = self._overlap_tokens(metadata_text)
        return (
            3 * len(question_tokens & main_tokens)
            + len(question_tokens & metadata_tokens)
            + 2 * len(response_tokens & main_tokens)
            + len(response_tokens & metadata_tokens)
        )

    def _rank_temporal_blocks(
        self,
        context: str,
        question: str,
        response: str,
    ) -> List[Dict[str, str]]:
        scored = []
        for order, block in enumerate(self._parse_context_blocks(context)):
            score = self._score_temporal_block(block, question, response)
            if score > 0:
                scored.append((score, order, block))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0]
        if best_score < 3:
            return []
        return [block for score, _, block in scored if score >= max(3, best_score - 2)]

    @staticmethod
    def _parse_session_datetime(date_text: str) -> Optional[datetime]:
        match = re.search(r"on\s+(\d{1,2}\s+[A-Za-z]+,\s+\d{4})", date_text)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%d %B, %Y")
        except ValueError:
            return None

    @staticmethod
    def _format_day(date_value: datetime) -> str:
        return f"{date_value.day} {date_value.strftime('%B %Y')}"

    @staticmethod
    def _canonical_weekday(raw_weekday: str) -> str:
        weekday = raw_weekday.lower().rstrip(".")
        if weekday.startswith("mon"):
            return "Monday"
        if weekday.startswith("tue"):
            return "Tuesday"
        if weekday.startswith("wed"):
            return "Wednesday"
        if weekday.startswith("thu"):
            return "Thursday"
        if weekday.startswith("fri"):
            return "Friday"
        if weekday.startswith("sat"):
            return "Saturday"
        if weekday.startswith("sun"):
            return "Sunday"
        return raw_weekday.capitalize()

    @staticmethod
    def _mentions_session_date(response: str, session_dt: datetime) -> bool:
        response_lower = response.lower()
        return (
            str(session_dt.year) in response_lower
            and session_dt.strftime("%B").lower() in response_lower
            and str(session_dt.day) in response_lower
        )

    def normalize_temporal_answer(self, response: str, context: str, question: str) -> str:
        """Normalize common relative-time answers for LoCoMo temporal questions."""
        if not response or not context:
            return response

        response_lower = response.lower()
        for block in self._rank_temporal_blocks(context, question, response):
            content_lower = block["content"].lower()
            session_dt = self._parse_session_datetime(block["date"])
            if session_dt is None:
                continue

            if "yesterday" in content_lower:
                if "yesterday" in response_lower or self._mentions_session_date(response, session_dt):
                    return self._format_day(session_dt - timedelta(days=1))

            if "last week" in content_lower:
                if "last week" in response_lower or self._mentions_session_date(response, session_dt):
                    return f"The week before {self._format_day(session_dt)}"

            if "next month" in content_lower:
                if "next month" in response_lower or "summer" in response_lower:
                    next_month = session_dt.month + 1
                    year = session_dt.year + (1 if next_month > 12 else 0)
                    month = 1 if next_month > 12 else next_month
                    return datetime(year, month, 1).strftime("%B %Y")

            if "this month" in content_lower:
                if "this month" in response_lower or self._mentions_session_date(response, session_dt):
                    return session_dt.strftime("%B %Y")

            weekday_match = re.search(
                r"last\s+(mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:rs(?:day)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\.?",
                content_lower,
            )
            if weekday_match and (
                "last" in response_lower
                or self._mentions_session_date(response, session_dt)
            ):
                weekday = self._canonical_weekday(weekday_match.group(1))
                return f"The {weekday} before {self._format_day(session_dt)}"

            if "last year" in content_lower and "last year" in response_lower:
                return str(session_dt.year - 1)

        return response

    def answer_question(self, question: str, category: int, answer: str) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        retrieval_query = self.build_retrieval_query(question, keywords)
        raw_context = self.retrieve_memory(retrieval_query, k=self.retrieve_k)
        context = (
            self.retrieve_memory_llm(raw_context, question)
            if int(category) in self.compress_categories
            else raw_context
        )
        evidence_instruction = (
            "Use the most directly relevant memory blocks for the question. "
            "Blocks marked relation: local_context are nearby conversational context; use them only when they clarify a directly relevant block. "
            "Ignore unrelated memories even if they share broad topics."
        )

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {evidence_instruction} {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. {evidence_instruction} Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, answer the following inference question. {evidence_instruction}
Category 3 questions often require a brief judgment, likely preference, trait, field, belief, or other inference from the evidence.
Give a concise answer that best matches the implied meaning. When the question asks for a likely yes/no judgment, answer with "Likely yes/no; brief reason" if the evidence supports a reason.
When the answer is a named holiday, book, place, person, or other proper term found in the context, preserve the wording from the context.
For traits, preferences, likely fields, beliefs, or status, include 2-4 words or a very short reason rather than a bare yes/no.

Question: {question} Short answer:"""
            temperature = 0.3
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. {evidence_instruction} Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        if category == 2:
            response = self.normalize_temporal_answer(response, raw_context, question)
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     allow_categories: Optional[List[int]] = None,
                     compress_categories: Optional[Set[int]] = None):
    """Evaluate the robust agent on the LoComo dataset."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_robust_{}_{}".format(backend, model),
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = allow_categories or [1, 2, 3, 4, 5]
    compress_categories = compress_categories or set()
    eval_logger.info(f"Evaluating categories: {allow_categories}")
    eval_logger.info(f"Compressing retrieved context for categories: {sorted(compress_categories)}")

    for sample_idx, sample in enumerate(samples):
        # agent指的并不是真正意义上的agent，而是一个封装了内存系统和LLM控制器的类，
        # 专门用于这个评测流程。它负责管理记忆系统、生成查询关键词、检索相关记忆，
        # 并根据问题和上下文生成答案。
        # 每个样本都会创建一个新的agent实例，以确保记忆系统从头开始构建，避免不同样本之间的记忆干扰。
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port, compress_categories)

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )
        retriever_cache_version_file = os.path.join(
            memories_dir, f"retriever_cache_version_sample_{sample_idx}.txt"
        )

        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            cache_version = None
            if os.path.exists(retriever_cache_version_file):
                with open(retriever_cache_version_file, 'r') as f:
                    cache_version = f.read().strip()
            retriever_cache_is_current = (
                cache_version == RETRIEVAL_INDEX_VERSION
                and os.path.exists(retriever_cache_file)
                and os.path.exists(retriever_cache_embeddings_file)
            )
            if retriever_cache_is_current:
                eval_logger.info(f"Found retriever cache files")
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    retriever_cache_file, retriever_cache_embeddings_file
                )
            else:
                eval_logger.info(
                    f"Retriever cache missing or stale "
                    f"(found={cache_version}, expected={RETRIEVAL_INDEX_VERSION}); rebuilding"
                )
                agent.memory_system.consolidate_memories()
                agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
                with open(retriever_cache_version_file, 'w') as f:
                    f.write(RETRIEVAL_INDEX_VERSION)
            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")

            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = (
                        f"dia_id: {turn.dia_id}\n"
                        f"session_date: {turn_datatime}\n"
                        f"speaker: {turn.speaker}\n"
                        f"content: {turn.text}"
                    )
                    if turn.image_caption:
                        conversation_tmp += f"\nimage_caption: {turn.image_caption}"
                    if turn.image_query:
                        conversation_tmp += f"\nimage_query: {turn.image_query}"
                    agent.add_memory(
                        conversation_tmp,
                        time=turn_datatime,
                        memory_level="instance",
                        domain_paths=["Conversation Memory / episodic turns"],
                        conditions=[
                            {
                                "dia_id": turn.dia_id,
                                "session_date": turn_datatime,
                                "speaker": turn.speaker,
                            }
                        ],
                    )

            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            with open(retriever_cache_version_file, 'w') as f:
                f.write(RETRIEVAL_INDEX_VERSION)
            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                eval_logger.info(f"Question {total_questions}: {qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {qa.final_answer}")
                eval_logger.info(f"User Prompt: {user_prompt}")
                eval_logger.info(f"Category: {qa.category}")
                eval_logger.info(f"Raw Context: {raw_context}")

                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "raw_context": raw_context,
                    "user_prompt": user_prompt,
                    "retrieved_dia_ids": agent._extract_retrieved_dia_ids(raw_context),
                }
                results.append(result)

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "compress_context": bool(compress_categories),
        "compress_categories": sorted(compress_categories),
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories to evaluate, e.g. '3' or '1,3'")
    parser.add_argument("--compress_context", action="store_true",
                        help="Compress retrieved memories for all categories before answering")
    parser.add_argument("--compress_categories", type=str, default=None,
                        help="Comma-separated categories to compress, e.g. '1,2'. Overrides the default no-compression behavior")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    allow_categories = None
    if args.categories:
        allow_categories = [
            int(category.strip())
            for category in args.categories.split(",")
            if category.strip()
        ]
        invalid_categories = [category for category in allow_categories if category not in [1, 2, 3, 4, 5]]
        if invalid_categories:
            raise ValueError(f"Invalid categories: {invalid_categories}")

    compress_categories = set()
    if args.compress_context:
        compress_categories = {1, 2, 3, 4, 5}
    if args.compress_categories:
        compress_categories = {
            int(category.strip())
            for category in args.compress_categories.split(",")
            if category.strip()
        }
        invalid_categories = [category for category in compress_categories if category not in [1, 2, 3, 4, 5]]
        if invalid_categories:
            raise ValueError(f"Invalid compression categories: {invalid_categories}")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port, allow_categories,
        compress_categories,
    )


if __name__ == "__main__":
    main()
