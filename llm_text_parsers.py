"""
Plain-text prompt templates, section-marker parsers, and validation logic
for the robust A-MEM system. Replaces JSON-schema LLM calls with plain-text
prompts that work with any LLM backend (Ollama, SGLang, OpenAI, etc.).
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing.

    Many models emit valid JSON even without strict mode, so we try that first
    for best-of-both-worlds compatibility.
    """
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items.

    Handles:
      - Bullet points (-, *, numbered)
      - Comma-separated values
      - One item per line
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = re.sub(r'^[\-\*\u2022]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        # If the line contains commas, split on them
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end).

    Args:
        text: Full LLM response
        marker: Section header to find (e.g. "KEYWORDS")
        next_markers: List of possible next section headers

    Returns:
        The text content of that section (may be empty string).
    """
    # Build a regex that finds the marker (case-insensitive) followed by a colon
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    # The first line of content may be on the same line as the marker
    first_line = match.group(1).strip()

    # Find where the next section starts
    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


MEMORY_RELATION_RESOLUTION_PROMPT = """You are a lifecycle manager for an agent memory system.
Decide how the new memory candidate should be written relative to existing candidate memories.

Important rules:
- Do not create update, replace, or conflict graph edges.
- If the new memory supersedes, conflicts with, or refines an old memory, choose a write-time consolidation action.
- Only choose create_separate_memory when the new memory should remain an independent active memory.
- Target is the memory index from the candidates below, or NONE.

Allowed actions:
- overwrite_existing: newer information clearly supersedes the target memory.
- refine_conditions: both memories remain valid under different contexts or conditions.
- append_detail: the new memory adds details to the target memory.
- create_separate_memory: the new memory belongs to a different entity, topic, or context.
- mark_candidate_uncertain: the new memory is temporary, ambiguous, or unreliable.

New memory:
- Content: {content}
- Context: {context}
- Keywords: {keywords}
- Tags: {tags}

Candidate memories:
{candidate_memories}

Respond using EXACTLY this format:
ACTION: <one allowed action>
TARGET: <candidate memory index or NONE>
RELATION: <same_entity|same_topic|semantic_related|elaborates|evidence_for|generalizes|co_used|none>
CONFIDENCE: <0.0 to 1.0>
REASON: <brief explanation>"""


MEMORY_REWRITE_PROMPT = """Rewrite an existing memory object at write time.
The rewritten memory must represent the current valid information. Old versions are kept only in revision history by the system, so do not include revision history in your answer.

Rewrite mode: {mode}
Resolution reason: {reason}

Existing memory:
- Content: {old_content}
- Context: {old_context}
- Conditions: {old_conditions}

New memory:
- Content: {new_content}
- Context: {new_context}

Respond using EXACTLY this format:
CONTENT: <rewritten current memory content>
CONDITIONS:
- context: <condition or scope>
  detail: <preference/fact/detail>
DOMAIN_PATHS: path1, path2
MEMORY_LEVEL: <instance|task|generalized>
CONFIDENCE: <0.0 to 1.0>
UPDATE_REASON: <brief reason for the rewrite>"""


DOMAIN_RERANK_PROMPT = """Choose the most appropriate memory domains for the text.
You are given embedding-retrieved candidate domains from a memory domain tree.
Select up to {max_domains} domains. Prefer the most specific domain path when possible.
You must select at least one domain from the candidate list.

Text:
{text}

Candidate domains:
{candidate_domains}

Respond using EXACTLY this format:
DOMAINS: domain path 1, domain path 2
REASON: <brief explanation>"""


# ---------------------------------------------------------------------------
# Parsers for each call site
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response.

    Returns:
        {"keywords": [...], "context": "...", "tags": [...]}
    """
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)

    # Validate / repair
    result = validate_analysis_result(result, content)
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response.

    Returns:
        {"decision": "NO_EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR|STRENGTHEN_AND_UPDATE",
         "reason": "..."}
    """
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        # Normalize common variants
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            # Try to infer from keywords
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Map JSON keys if we got JSON
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response.

    Returns:
        {"connections": [int, ...], "tags": [str, ...]}
    """
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        # Parse connections as integers
        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    # Map from JSON keys if needed
    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [int(x) for x in result.get("suggested_connections", []) if isinstance(x, (int, float))]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response.

    Returns:
        [{"context": "...", "tags": [...]}, ...] — one per neighbor
    """
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            # Try to find NEIGHBOR i: block
            # Look for "NEIGHBOR i:" or "NEIGHBOR i\n"
            pattern = re.compile(
                rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE
            )
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            # Find the end of this neighbor block (next NEIGHBOR or end)
            next_pattern = re.compile(
                rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE
            )
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            tags = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    # Try JSON first
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


def parse_memory_relation_resolution(response: str) -> Dict[str, Any]:
    """Parse a write-time memory relation decision."""
    valid_actions = {
        "overwrite_existing",
        "refine_conditions",
        "append_detail",
        "create_separate_memory",
        "mark_candidate_uncertain",
    }
    valid_relations = {
        "semantic_related",
        "same_topic",
        "same_entity",
        "evidence_for",
        "generalizes",
        "elaborates",
        "co_used",
        "none",
    }

    def _section_parse(resp: str) -> Dict[str, Any]:
        action = _extract_section(resp, "ACTION", ["TARGET", "RELATION", "CONFIDENCE", "REASON"])
        target = _extract_section(resp, "TARGET", ["ACTION", "RELATION", "CONFIDENCE", "REASON"])
        relation = _extract_section(resp, "RELATION", ["ACTION", "TARGET", "CONFIDENCE", "REASON"])
        confidence_text = _extract_section(resp, "CONFIDENCE", ["ACTION", "TARGET", "RELATION", "REASON"])
        reason = _extract_section(resp, "REASON", ["ACTION", "TARGET", "RELATION", "CONFIDENCE"])
        return {
            "action": action.strip(),
            "target": target.strip(),
            "relation": relation.strip(),
            "confidence": confidence_text.strip(),
            "reason": reason.strip(),
        }

    result = parse_with_json_fallback(response, _section_parse)
    action = str(result.get("action") or result.get("ACTION") or "").strip().lower()
    action = action.replace(" ", "_").replace("-", "_")
    if action not in valid_actions:
        action = "create_separate_memory"

    raw_target = result.get("target", result.get("TARGET", "NONE"))
    target = None
    if raw_target is not None and str(raw_target).strip().upper() != "NONE":
        match = re.search(r"-?\d+", str(raw_target))
        if match:
            target = int(match.group(0))

    relation = str(result.get("relation") or result.get("RELATION") or "semantic_related").strip().lower()
    relation = relation.replace(" ", "_").replace("-", "_")
    if relation not in valid_relations:
        relation = "semantic_related"

    try:
        confidence = float(result.get("confidence", result.get("CONFIDENCE", 0.6)))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    return {
        "action": action,
        "target": target,
        "relation": relation,
        "confidence": confidence,
        "reason": str(result.get("reason") or result.get("REASON") or "").strip(),
    }


def parse_memory_rewrite(response: str) -> Dict[str, Any]:
    """Parse a rewritten memory object produced by the lifecycle manager."""
    def _section_parse(resp: str) -> Dict[str, Any]:
        content = _extract_section(resp, "CONTENT", ["CONDITIONS", "DOMAIN_PATHS", "MEMORY_LEVEL", "CONFIDENCE", "UPDATE_REASON"])
        conditions_text = _extract_section(resp, "CONDITIONS", ["DOMAIN_PATHS", "MEMORY_LEVEL", "CONFIDENCE", "UPDATE_REASON"])
        domain_paths = _extract_section(resp, "DOMAIN_PATHS", ["CONTENT", "CONDITIONS", "MEMORY_LEVEL", "CONFIDENCE", "UPDATE_REASON"])
        memory_level = _extract_section(resp, "MEMORY_LEVEL", ["CONTENT", "CONDITIONS", "DOMAIN_PATHS", "CONFIDENCE", "UPDATE_REASON"])
        confidence = _extract_section(resp, "CONFIDENCE", ["CONTENT", "CONDITIONS", "DOMAIN_PATHS", "MEMORY_LEVEL", "UPDATE_REASON"])
        update_reason = _extract_section(resp, "UPDATE_REASON", ["CONTENT", "CONDITIONS", "DOMAIN_PATHS", "MEMORY_LEVEL", "CONFIDENCE"])
        return {
            "content": content.strip(),
            "conditions": conditions_text.strip(),
            "domain_paths": domain_paths.strip(),
            "memory_level": memory_level.strip(),
            "confidence": confidence.strip(),
            "update_reason": update_reason.strip(),
        }

    result = parse_with_json_fallback(response, _section_parse)

    content = str(result.get("content") or result.get("CONTENT") or "").strip()
    raw_conditions = result.get("conditions", result.get("CONDITIONS", []))
    if isinstance(raw_conditions, list):
        conditions = [c for c in raw_conditions if c]
    else:
        conditions = []
        current = {}
        for line in str(raw_conditions).splitlines():
            line = re.sub(r"^[\-\*\u2022]\s*", "", line.strip())
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key == "context" and current:
                    conditions.append(current)
                    current = {}
                current[key] = value
        if current:
            conditions.append(current)

    raw_paths = result.get("domain_paths", result.get("DOMAIN_PATHS", []))
    domain_paths = raw_paths if isinstance(raw_paths, list) else _parse_list_items(str(raw_paths))

    memory_level = str(result.get("memory_level", result.get("MEMORY_LEVEL", "instance"))).strip().lower()
    if memory_level not in {"instance", "task", "generalized"}:
        memory_level = "instance"

    try:
        confidence = float(result.get("confidence", result.get("CONFIDENCE", 0.8)))
    except (TypeError, ValueError):
        confidence = 0.8
    confidence = max(0.0, min(1.0, confidence))

    return {
        "content": content,
        "conditions": conditions,
        "domain_paths": domain_paths,
        "memory_level": memory_level,
        "confidence": confidence,
        "update_reason": str(result.get("update_reason") or result.get("UPDATE_REASON") or "").strip(),
    }


def parse_domain_rerank(response: str, valid_domains: List[str], max_domains: int = 3) -> Dict[str, Any]:
    """Parse an LLM domain rerank response and keep only valid candidate domains."""
    def _section_parse(resp: str) -> Dict[str, Any]:
        domains_text = _extract_section(resp, "DOMAINS", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DOMAINS"])
        return {
            "domains": _parse_list_items(domains_text),
            "reason": reason_text.strip(),
        }

    result = parse_with_json_fallback(response, _section_parse)
    raw_domains = result.get("domains", result.get("DOMAINS", []))
    if isinstance(raw_domains, str):
        raw_domains = _parse_list_items(raw_domains)

    valid_by_lower = {domain.lower(): domain for domain in valid_domains}
    selected: List[str] = []
    for domain in raw_domains:
        domain_text = str(domain).strip()
        matched = valid_by_lower.get(domain_text.lower())
        if matched and matched not in selected:
            selected.append(matched)
        if len(selected) >= max_domains:
            break

    if not selected and valid_domains:
        selected = [valid_domains[0]]

    return {
        "domains": selected[:max_domains],
        "reason": str(result.get("reason") or result.get("REASON") or "").strip(),
    }


def parse_plain_text_answer(response: str) -> str:
    """Parse a plain-text answer response (for QA evaluation).

    If the model returned JSON with an "answer" field, extract it.
    Otherwise return the raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_relevant_parts(response: str) -> str:
    """Parse retrieve_memory_llm response.

    If JSON with "relevant_parts", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "relevant_parts" in data:
            return str(data["relevant_parts"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_keywords_response(response: str) -> str:
    """Parse generate_query_llm response.

    If JSON with "keywords", extract it. Otherwise return raw text.
    """
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result.

    - If keywords is empty, extract capitalized words / nouns heuristically.
    - If context is empty, use the first sentence of content.
    - If tags is empty, derive from keywords.
    """
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # Ensure lists
    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    # Repair empty keywords from content
    if not keywords and content:
        keywords = _heuristic_keywords(content)

    # Repair empty context from content
    if not context and content:
        context = _heuristic_context(content)

    # Repair empty tags from keywords
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    # Remove common stop words and extract significant words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    # Prefer capitalized words (likely proper nouns / key terms)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    # Take the first sentence (up to period, question mark, or exclamation)
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    # Fallback: first 200 chars
    return content[:200].strip()
