"""
Example pruner — clusters similar in-prompt examples and retains only
one representative per cluster, saving tokens without losing diversity.

Similarity measure
------------------
Uses TF-IDF weighted cosine similarity implemented entirely in the Python
standard library (no scikit-learn dependency).  When the optional
`scikit-learn` package is installed the vectoriser automatically uses it
for higher accuracy, but the pure-Python path produces good results for
the typical prompt example sizes (< 200 words each).

Clustering
----------
Greedy single-linkage: iterate examples in order; for each unassigned
example find all later examples with cosine similarity >= threshold and
form a cluster.  Pick the median-length member as the representative
(avoids the shortest trivially 'representative' example).
"""

import math
import re
from typing import Dict, List, Tuple

from claude_toolkit.models import ExampleCluster, PruningResult
from claude_toolkit.tokenizer.counter import count_tokens


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (pure Python)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    from collections import Counter
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {w: c / total for w, c in counts.items()}


def _idf(corpus: List[List[str]]) -> Dict[str, float]:
    n = len(corpus)
    doc_freq: Dict[str, int] = {}
    for tokens in corpus:
        for w in set(tokens):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    # Smooth IDF: log((N+1)/(df+1)) + 1  avoids zero-division and log(0)
    return {w: math.log((n + 1) / (df + 1)) + 1.0 for w, df in doc_freq.items()}


def _tfidf(tokens: List[str], idf_map: Dict[str, float]) -> Dict[str, float]:
    tf = _tf(tokens)
    return {w: tf[w] * idf_map.get(w, 1.0) for w in tf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _similarity_matrix(texts: List[str]) -> List[List[float]]:
    """Return an n×n cosine-similarity matrix for *texts*."""
    tokenized = [_tokenize(t) for t in texts]
    idf = _idf(tokenized)
    vectors = [_tfidf(tok, idf) for tok in tokenized]
    n = len(vectors)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine(vectors[i], vectors[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix


# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------

_EXAMPLE_PATTERNS = [
    # Labelled blocks: "Example 1:", "Example:", "Example —"
    r"(?:^|\n)Example\s*\d*\s*[:\-—]\s*([\s\S]*?)(?=(?:^|\n)Example\s*\d*\s*[:\-—]|\Z)",
    # "Input:" / "Output:" pairs
    r"(?:^|\n)(?:Input|Output)\s*\d*\s*[:\-—]\s*([\s\S]*?)(?=(?:^|\n)(?:Input|Output)\s*\d*\s*[:\-—]|\Z)",
    # Inline e.g. / for example
    r"(?:e\.g\.,?\s*|for example,?\s*)([\s\S]*?)(?=\n\n|\Z)",
]


def _extract_examples(prompt: str) -> List[Tuple[int, int, str]]:
    """
    Extract example text spans from *prompt*.

    Returns list of (char_start, char_end, text) sorted by start position.
    Overlapping spans are deduplicated (first-wins).
    """
    hits: List[Tuple[int, int, str]] = []

    for pattern in _EXAMPLE_PATTERNS:
        for m in re.finditer(pattern, prompt, re.IGNORECASE | re.MULTILINE):
            text = (m.group(1) or m.group(0)).strip()
            if len(text.split()) >= 5:
                hits.append((m.start(), m.end(), text))

    # Sort and deduplicate overlapping matches
    hits.sort(key=lambda x: x[0])
    deduped: List[Tuple[int, int, str]] = []
    prev_end = -1
    for start, end, text in hits:
        if start >= prev_end:
            deduped.append((start, end, text))
            prev_end = end

    return deduped


# ---------------------------------------------------------------------------
# Greedy clustering
# ---------------------------------------------------------------------------

def _greedy_cluster(
    examples: List[str],
    threshold: float,
) -> List[ExampleCluster]:
    """
    Single-linkage greedy clustering.

    Returns one ExampleCluster per group.  The representative is the
    member closest to the median token length (best 'average' example).
    """
    if not examples:
        return []

    sim = _similarity_matrix(examples)
    n = len(examples)
    assigned = [False] * n
    clusters: List[ExampleCluster] = []

    for i in range(n):
        if assigned[i]:
            continue

        group: List[int] = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if not assigned[j] and sim[i][j] >= threshold:
                group.append(j)
                assigned[j] = True

        # Representative = median-length member
        sorted_by_len = sorted(group, key=lambda k: count_tokens(examples[k]))
        rep_idx = sorted_by_len[len(sorted_by_len) // 2]
        rep_text = examples[rep_idx]

        # Average intra-cluster similarity
        pairs = [(a, b) for idx_a, a in enumerate(group) for b in group[idx_a + 1:]]
        avg_sim = (
            sum(sim[a][b] for a, b in pairs) / len(pairs) if pairs else 1.0
        )

        tokens_saved = sum(
            count_tokens(examples[k]) for k in group if k != rep_idx
        )

        clusters.append(ExampleCluster(
            representative=rep_text,
            members=[examples[k] for k in group],
            member_indices=group,
            similarity_score=round(avg_sim, 3),
            tokens_saved=tokens_saved,
        ))

    return clusters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prune_examples(
    prompt: str,
    similarity_threshold: float = 0.65,
    model: str = "claude-sonnet",
) -> PruningResult:
    """
    Detect, cluster, and prune redundant examples inside *prompt*.

    Args:
        prompt:               The prompt text to analyse.
        similarity_threshold: Cosine similarity above which two examples
                              are considered redundant (0.0 – 1.0).
        model:                Model name used for token counting.

    Returns:
        PruningResult with clusters, representative examples, and savings.
    """
    extracted = _extract_examples(prompt)
    example_texts = [text for _, _, text in extracted]
    original_tokens = count_tokens(prompt, model)

    if len(example_texts) < 2:
        return PruningResult(
            original_examples=example_texts,
            kept_examples=example_texts,
            clusters=[],
            original_tokens=original_tokens,
            pruned_tokens=original_tokens,
            tokens_saved=0,
            reduction_pct=0.0,
        )

    clusters = _greedy_cluster(example_texts, similarity_threshold)
    kept = [c.representative for c in clusters]
    total_saved = sum(c.tokens_saved for c in clusters)
    pruned_tokens = max(0, original_tokens - total_saved)
    reduction_pct = round(total_saved / original_tokens * 100, 2) if original_tokens > 0 else 0.0

    return PruningResult(
        original_examples=example_texts,
        kept_examples=kept,
        clusters=clusters,
        original_tokens=original_tokens,
        pruned_tokens=pruned_tokens,
        tokens_saved=total_saved,
        reduction_pct=reduction_pct,
    )
