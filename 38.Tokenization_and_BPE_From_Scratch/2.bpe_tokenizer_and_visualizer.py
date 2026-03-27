from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def pretokenize(text: str) -> List[str]:
    text = text.lower().strip()
    # Keep words and simple punctuation blocks.
    return re.findall(r"[a-z0-9]+|[^\w\s]", text)


def init_vocab(words: List[str]) -> Counter:
    # Word represented as tuple(chars + end marker)
    vocab = Counter()
    for w in words:
        vocab[tuple(list(w) + ["</w>"])] += 1
    return vocab


def get_pair_counts(vocab: Counter) -> Counter:
    pair_counts = Counter()
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += freq
    return pair_counts


def merge_pair(vocab: Counter, pair: Tuple[str, str]) -> Counter:
    new_vocab = Counter()
    a, b = pair
    merged = a + b

    for tokens, freq in vocab.items():
        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        new_vocab[tuple(out)] += freq

    return new_vocab


def train_bpe(corpus: List[str], num_merges: int = 50):
    words = []
    for line in corpus:
        words.extend([t for t in pretokenize(line) if re.match(r"[a-z0-9]+", t)])

    vocab = init_vocab(words)
    merges = []
    merge_freqs = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break
        vocab = merge_pair(vocab, best_pair)
        merges.append(best_pair)
        merge_freqs.append(best_count)

    return vocab, merges, merge_freqs


def build_token_set(vocab: Counter) -> Counter:
    c = Counter()
    for token_tuple, freq in vocab.items():
        for t in token_tuple:
            c[t] += freq
    return c


def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    tokens = list(word) + ["</w>"]
    merge_map = {p: i for i, p in enumerate(merges)}

    while True:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        candidates = [(merge_map[p], p) for p in pairs if p in merge_map]
        if not candidates:
            break
        _, best = min(candidates)

        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                out.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out

    return tokens


def ascii_bar(values: List[Tuple[str, int]], width: int = 40) -> None:
    if not values:
        return
    max_v = max(v for _, v in values)
    for k, v in values:
        n = int((v / max_v) * width) if max_v else 0
        print(f"{k[:16]:>16} | {'#' * n} {v}")


def maybe_plot_bar(labels: List[str], vals: List[int], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; showing ASCII chart instead")
        ascii_bar(list(zip(labels, vals)))
        return

    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals)
    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.show()


def compression_ratio(raw_words: List[str], merges: List[Tuple[str, str]]) -> float:
    before = sum(len(w) + 1 for w in raw_words)  # +1 for end marker
    after = sum(len(encode_word(w, merges)) for w in raw_words)
    return after / max(1, before)


if __name__ == "__main__":
    corpus = [
        "Transformers need good tokenization for efficient training.",
        "Tokenization quality changes sequence length and memory.",
        "BPE merges frequent pairs into larger subword units.",
        "Pure python tokenizer training helps understanding internals.",
        "Kaggle and Hugging Face workflows both depend on clean text pipelines.",
    ]

    vocab, merges, merge_freqs = train_bpe(corpus, num_merges=60)
    token_counts = build_token_set(vocab)

    print("Total merges learned:", len(merges))
    print("Top 10 merges:", merges[:10])

    top_tokens = token_counts.most_common(12)
    labels = [k for k, _ in top_tokens]
    vals = [v for _, v in top_tokens]
    maybe_plot_bar(labels, vals, "Top Token Frequencies (BPE)")

    words = []
    for line in corpus:
        words.extend([t for t in pretokenize(line) if re.match(r"[a-z0-9]+", t)])
    ratio = compression_ratio(words, merges)
    print("Compression ratio (after/before):", round(ratio, 4))

    sample = "tokenization"
    print("Encoded 'tokenization':", encode_word(sample, merges))
