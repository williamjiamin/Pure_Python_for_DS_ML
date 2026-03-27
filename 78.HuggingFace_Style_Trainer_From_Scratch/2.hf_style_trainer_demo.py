from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy_prob(prob: float) -> float:
    p = min(1 - 1e-12, max(1e-12, prob))
    return -math.log(p)


def make_sentence(label: int, rng: random.Random) -> str:
    positive = ["great", "excellent", "amazing", "love", "clean", "fast", "helpful", "smooth"]
    negative = ["bad", "terrible", "awful", "hate", "slow", "broken", "dirty", "buggy"]
    neutral = ["product", "service", "support", "update", "delivery", "price", "experience", "team"]

    sent_tokens = []

    main_pool = positive if label == 1 else negative
    other_pool = negative if label == 1 else positive

    n_main = rng.randint(2, 4)
    n_neutral = rng.randint(2, 4)
    n_noise = 1 if rng.random() < 0.22 else 0

    for _ in range(n_main):
        sent_tokens.append(rng.choice(main_pool))
    for _ in range(n_neutral):
        sent_tokens.append(rng.choice(neutral))
    for _ in range(n_noise):
        sent_tokens.append(rng.choice(other_pool))

    rng.shuffle(sent_tokens)
    return " ".join(sent_tokens)


def build_dataset(n: int = 3600, seed: int = 42):
    rng = random.Random(seed)
    texts: List[str] = []
    labels: List[int] = []

    for _ in range(n):
        y = 1 if rng.random() < 0.5 else 0
        text = make_sentence(y, rng)

        # Label noise to force realistic trainer behavior.
        if rng.random() < 0.08:
            y = 1 - y

        texts.append(text)
        labels.append(y)

    cut = int(0.78 * n)
    return (
        texts[:cut],
        labels[:cut],
        texts[cut:],
        labels[cut:],
    )


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for tok, c in sorted(freq.items()):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_texts(texts: List[str], vocab: Dict[str, int]) -> List[List[int]]:
    out = []
    for t in texts:
        ids = [vocab.get(tok, 1) for tok in tokenize(t)]
        out.append(ids if ids else [1])
    return out


def init_matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.05):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def zeros_like_matrix(mat: List[List[float]]):
    return [[0.0 for _ in row] for row in mat]


def zeros_like_vector(vec: List[float]):
    return [0.0 for _ in vec]


def predict_single(ids: List[int], emb: List[List[float]], W: List[List[float]], b: List[float]):
    d = len(W)
    rep = [0.0] * d
    inv = 1.0 / len(ids)
    for idx in ids:
        e = emb[idx]
        for j in range(d):
            rep[j] += e[j] * inv

    logits = [b[0], b[1]]
    for j in range(d):
        logits[0] += rep[j] * W[j][0]
        logits[1] += rep[j] * W[j][1]

    probs = softmax(logits)
    return rep, logits, probs


def lr_schedule(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step <= warmup_steps:
        frac = step / max(1, warmup_steps)
        return lr_min + frac * (lr_max - lr_min)

    frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
    return lr_min + cosine * (lr_max - lr_min)


def global_grad_norm(
    g_emb: List[List[float]],
    g_W: List[List[float]],
    g_b: List[float],
) -> float:
    s = 0.0
    for row in g_emb:
        for v in row:
            s += v * v
    for row in g_W:
        for v in row:
            s += v * v
    for v in g_b:
        s += v * v
    return math.sqrt(s)


def apply_clip(
    g_emb: List[List[float]],
    g_W: List[List[float]],
    g_b: List[float],
    clip_norm: float,
):
    norm = global_grad_norm(g_emb, g_W, g_b)
    if norm <= clip_norm:
        return norm

    scale = clip_norm / (norm + 1e-12)
    for i in range(len(g_emb)):
        row = g_emb[i]
        for j in range(len(row)):
            row[j] *= scale
    for i in range(len(g_W)):
        row = g_W[i]
        for j in range(len(row)):
            row[j] *= scale
    for j in range(len(g_b)):
        g_b[j] *= scale

    return norm


def adamw_update_matrix(
    param: List[List[float]],
    grad: List[List[float]],
    m: List[List[float]],
    v: List[List[float]],
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
):
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t

    for i in range(len(param)):
        for j in range(len(param[i])):
            g = grad[i][j]
            m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g
            v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * g * g

            m_hat = m[i][j] / bc1
            v_hat = v[i][j] / bc2

            param[i][j] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + weight_decay * param[i][j])


def adamw_update_vector(
    param: List[float],
    grad: List[float],
    m: List[float],
    v: List[float],
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
):
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t

    for i in range(len(param)):
        g = grad[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * g
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g

        m_hat = m[i] / bc1
        v_hat = v[i] / bc2

        param[i] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + weight_decay * param[i])


def evaluate(
    X: List[List[int]],
    y: List[int],
    emb: List[List[float]],
    W: List[List[float]],
    b: List[float],
):
    losses = []
    preds = []
    probs = []

    for ids, yi in zip(X, y):
        _, _, p = predict_single(ids, emb, W, b)
        losses.append(cross_entropy_prob(p[yi]))
        probs.append(p[1])
        preds.append(1 if p[1] >= 0.5 else 0)

    tp = sum(1 for yt, yp in zip(y, preds) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y, preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y, preds) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y, preds) if yt == 0 and yp == 0)

    acc = (tp + tn) / len(y)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "loss": sum(losses) / len(losses),
        "acc": acc,
        "f1": f1,
        "cm": [[tn, fp], [fn, tp]],
        "probs": probs,
        "preds": preds,
    }


def copy_matrix(mat: List[List[float]]):
    return [row[:] for row in mat]


def copy_vector(vec: List[float]):
    return vec[:]


def top_errors(
    texts: List[str],
    y: List[int],
    preds: List[int],
    probs: List[float],
    k: int = 8,
):
    rows = []
    for t, yt, yp, p1 in zip(texts, y, preds, probs):
        if yt == yp:
            continue
        conf = p1 if yp == 1 else (1 - p1)
        rows.append((conf, yt, yp, t))

    rows.sort(reverse=True, key=lambda x: x[0])
    return rows[:k]


def maybe_plot(history: Dict[str, List[float]], cm: List[List[int]]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not installed. Skipping plots.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    steps = list(range(1, len(history["lr"]) + 1))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))

    axs[0].plot(epochs, history["train_loss"], label="train", color="#1d4ed8")
    axs[0].plot(epochs, history["valid_loss"], label="valid", color="#f59e0b")
    axs[0].set_title("Loss by Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-entropy")
    axs[0].legend()

    axs[1].plot(steps, history["lr"], color="#059669")
    axs[1].set_title("Warmup + Cosine LR")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Learning rate")

    axs[2].imshow(cm, cmap="Blues")
    axs[2].set_title("Validation Confusion Matrix")
    axs[2].set_xticks([0, 1])
    axs[2].set_yticks([0, 1])
    axs[2].set_xticklabels(["pred_0", "pred_1"])
    axs[2].set_yticklabels(["true_0", "true_1"])
    for i in range(2):
        for j in range(2):
            axs[2].text(j, i, str(cm[i][j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_texts, train_y, valid_texts, valid_y = build_dataset(n=3600, seed=12)

    vocab = build_vocab(train_texts, min_freq=1)
    X_train = encode_texts(train_texts, vocab)
    X_valid = encode_texts(valid_texts, vocab)

    rng = random.Random(3)
    vocab_size = len(vocab)
    hidden_dim = 24

    emb = init_matrix(vocab_size, hidden_dim, rng, scale=0.08)
    W = init_matrix(hidden_dim, 2, rng, scale=0.08)
    b = [0.0, 0.0]

    m_emb = zeros_like_matrix(emb)
    v_emb = zeros_like_matrix(emb)
    m_W = zeros_like_matrix(W)
    v_W = zeros_like_matrix(W)
    m_b = zeros_like_vector(b)
    v_b = zeros_like_vector(b)

    batch_size = 64
    epochs = 18
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wd_weights = 0.01
    clip_norm = 1.2

    lr_max = 0.035
    lr_min = 0.002

    steps_per_epoch = math.ceil(len(train_y) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.08 * total_steps))

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "valid_f1": [],
        "lr": [],
    }

    best_valid = float("inf")
    best_state = None
    patience = 4
    wait = 0
    step = 0

    for epoch in range(1, epochs + 1):
        idx = list(range(len(train_y)))
        random.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            g_emb = zeros_like_matrix(emb)
            g_W = zeros_like_matrix(W)
            g_b = zeros_like_vector(b)

            for i in batch_idx:
                ids = X_train[i]
                yi = train_y[i]

                rep, _, probs = predict_single(ids, emb, W, b)
                dlogits = [probs[0], probs[1]]
                dlogits[yi] -= 1.0

                for j in range(hidden_dim):
                    g_W[j][0] += rep[j] * dlogits[0]
                    g_W[j][1] += rep[j] * dlogits[1]
                g_b[0] += dlogits[0]
                g_b[1] += dlogits[1]

                drep = [
                    W[j][0] * dlogits[0] + W[j][1] * dlogits[1]
                    for j in range(hidden_dim)
                ]
                inv = 1.0 / len(ids)
                for tok in ids:
                    for j in range(hidden_dim):
                        g_emb[tok][j] += drep[j] * inv

            scale = 1.0 / max(1, len(batch_idx))
            for i in range(vocab_size):
                row = g_emb[i]
                for j in range(hidden_dim):
                    row[j] *= scale
            for i in range(hidden_dim):
                row = g_W[i]
                for j in range(2):
                    row[j] *= scale
            for j in range(2):
                g_b[j] *= scale

            apply_clip(g_emb, g_W, g_b, clip_norm=clip_norm)

            step += 1
            lr = lr_schedule(step, total_steps, warmup_steps, lr_max, lr_min)
            history["lr"].append(lr)

            adamw_update_matrix(emb, g_emb, m_emb, v_emb, step, lr, beta1, beta2, eps, wd_weights)
            adamw_update_matrix(W, g_W, m_W, v_W, step, lr, beta1, beta2, eps, wd_weights)
            adamw_update_vector(b, g_b, m_b, v_b, step, lr, beta1, beta2, eps, 0.0)

        train_metrics = evaluate(X_train, train_y, emb, W, b)
        valid_metrics = evaluate(X_valid, valid_y, emb, W, b)

        history["train_loss"].append(train_metrics["loss"])
        history["valid_loss"].append(valid_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["valid_acc"].append(valid_metrics["acc"])
        history["valid_f1"].append(valid_metrics["f1"])

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['acc']:.4f} valid_f1={valid_metrics['f1']:.4f}"
        )

        if valid_metrics["loss"] < best_valid - 1e-4:
            best_valid = valid_metrics["loss"]
            best_state = {
                "emb": copy_matrix(emb),
                "W": copy_matrix(W),
                "b": copy_vector(b),
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        emb = best_state["emb"]
        W = best_state["W"]
        b = best_state["b"]

    final_train = evaluate(X_train, train_y, emb, W, b)
    final_valid = evaluate(X_valid, valid_y, emb, W, b)

    print("Final train metrics:", {k: round(v, 5) for k, v in final_train.items() if k in ("loss", "acc", "f1")})
    print("Final valid metrics:", {k: round(v, 5) for k, v in final_valid.items() if k in ("loss", "acc", "f1")})
    print("Validation confusion matrix [ [TN, FP], [FN, TP] ]:", final_valid["cm"])

    hard_errors = top_errors(valid_texts, valid_y, final_valid["preds"], final_valid["probs"], k=8)
    print("Top confident errors:")
    for conf, yt, yp, txt in hard_errors:
        print({"confidence": round(conf, 4), "true": yt, "pred": yp, "text": txt})

    maybe_plot(history, final_valid["cm"])
