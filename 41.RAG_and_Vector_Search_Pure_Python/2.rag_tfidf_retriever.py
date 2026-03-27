from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(docs: List[str]) -> List[str]:
    s = set()
    for d in docs:
        s.update(tokenize(d))
    return sorted(s)


def tf(tokens: List[str]) -> Dict[str, float]:
    c = Counter(tokens)
    n = len(tokens)
    return {k: v / n for k, v in c.items()}


def idf(docs: List[str], vocab: List[str]) -> Dict[str, float]:
    N = len(docs)
    out = {}
    for t in vocab:
        df = sum(1 for d in docs if t in set(tokenize(d)))
        out[t] = math.log((N + 1) / (df + 1)) + 1.0
    return out


def to_vec(doc: str, vocab: List[str], idf_map: Dict[str, float]) -> List[float]:
    tfs = tf(tokenize(doc))
    return [tfs.get(tok, 0.0) * idf_map[tok] for tok in vocab]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query: str, docs: List[str], vectors: List[List[float]], vocab: List[str], idf_map: Dict[str, float], k: int = 3):
    qv = to_vec(query, vocab, idf_map)
    scored = [(cosine(qv, dv), i, docs[i]) for i, dv in enumerate(vectors)]
    scored.sort(reverse=True)
    return scored[:k]


def extractive_answer(query: str, top_docs: List[Tuple[float, int, str]]) -> str:
    q_tokens = set(tokenize(query))
    best_sentence = ""
    best_overlap = -1

    for _, _, doc in top_docs:
        sentences = re.split(r"[.!?]\s+", doc)
        for s in sentences:
            st = set(tokenize(s))
            overlap = len(q_tokens & st)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

    return best_sentence.strip()


def maybe_plot_scores(scores: List[float], labels: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Retrieval scores (ASCII):")
        m = max(scores) if scores else 1.0
        for l, s in zip(labels, scores):
            n = int((s / m) * 30) if m else 0
            print(f"{l:>8} | {'#' * n} {s:.4f}")
        return

    plt.figure(figsize=(7, 4))
    plt.bar(labels, scores)
    plt.title("Top-K Retrieval Scores")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    docs = [
        "Gradient boosting builds models sequentially by fitting residuals.",
        "Random forest reduces variance by averaging many decorrelated trees.",
        "Transformers rely on self-attention and tokenization quality.",
        "LoRA fine-tuning updates low-rank adapter matrices while freezing base weights.",
        "RAG systems retrieve external context before generating grounded answers.",
    ]

    vocab = build_vocab(docs)
    idf_map = idf(docs, vocab)
    vectors = [to_vec(d, vocab, idf_map) for d in docs]

    query = "How does LoRA reduce training cost?"
    top = retrieve(query, docs, vectors, vocab, idf_map, k=3)

    print("Query:", query)
    for rank, (score, idx, doc) in enumerate(top, 1):
        print(f"{rank}. score={score:.4f} doc#{idx}: {doc}")

    ans = extractive_answer(query, top)
    print("Grounded answer:", ans)

    maybe_plot_scores([s for s, _, _ in top], [f"doc{idx}" for _, idx, _ in top])
