from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tok(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.N = len(docs)

        self.df = Counter()
        self.tf = []
        for tokens in self.doc_tokens:
            c = Counter(tokens)
            self.tf.append(c)
            for term in c.keys():
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score_doc(self, query: str, idx: int) -> float:
        q_terms = tok(query)
        c = self.tf[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for t in q_terms:
            f = c.get(t, 0)
            if f == 0:
                continue
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def explain(self, query: str, idx: int) -> Dict[str, float]:
        q_terms = tok(query)
        c = self.tf[idx]
        dl = self.doc_len[idx]
        out = {}
        for t in q_terms:
            f = c.get(t, 0)
            if f == 0:
                out[t] = 0.0
                continue
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            out[t] = idf * (f * (self.k1 + 1)) / denom
        return out

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[float, int, str]]:
        scored = [(self.score_doc(query, i), i, self.docs[i]) for i in range(self.N)]
        scored.sort(reverse=True)
        return scored[:top_k]


def maybe_plot_contrib(contrib: Dict[str, float], title: str):
    terms = list(contrib.keys())
    vals = [contrib[t] for t in terms]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(title)
        for t, v in contrib.items():
            n = int(v * 25)
            print(f"{t:>12} | {'#' * max(0, n)} {v:.4f}")
        return

    plt.figure(figsize=(7, 4))
    plt.bar(terms, vals)
    plt.title(title)
    plt.ylabel("BM25 term contribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    docs = [
        "LoRA adapters reduce trainable parameters during fine tuning.",
        "BM25 is a sparse lexical retrieval baseline for ranking.",
        "Hybrid retrieval combines BM25 and dense embeddings.",
        "Q learning and SARSA are temporal difference reinforcement methods.",
        "Kaggle feature engineering requires strict leakage control.",
    ]

    query = "bm25 ranking retrieval"
    bm25 = BM25(docs, k1=1.5, b=0.75)
    top = bm25.rank(query, top_k=3)

    print("Query:", query)
    for r, (s, idx, doc) in enumerate(top, 1):
        print(f"{r}. score={s:.4f} doc#{idx}: {doc}")

    best_idx = top[0][1]
    contrib = bm25.explain(query, best_idx)
    maybe_plot_contrib(contrib, title=f"Term Contributions for doc#{best_idx}")
