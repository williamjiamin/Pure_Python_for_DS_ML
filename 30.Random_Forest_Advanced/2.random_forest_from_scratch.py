from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[Any] = None


def gini_impurity(labels: List[Any]) -> float:
    counts = Counter(labels)
    n = len(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def split_rows(rows: List[List[float]], y: List[Any], feature: int, threshold: float):
    left_rows, right_rows = [], []
    left_y, right_y = [], []
    for row, label in zip(rows, y):
        if row[feature] <= threshold:
            left_rows.append(row)
            left_y.append(label)
        else:
            right_rows.append(row)
            right_y.append(label)
    return left_rows, right_rows, left_y, right_y


def majority_class(labels: List[Any]) -> Any:
    return Counter(labels).most_common(1)[0][0]


def best_split(
    rows: List[List[float]],
    y: List[Any],
    feature_indices: List[int],
) -> Tuple[Optional[int], Optional[float], float]:
    best_feature, best_threshold = None, None
    best_score = float("inf")

    for feature in feature_indices:
        thresholds = sorted(set(row[feature] for row in rows))
        for threshold in thresholds:
            _, _, left_y, right_y = split_rows(rows, y, feature, threshold)
            if not left_y or not right_y:
                continue
            w_left = len(left_y) / len(y)
            w_right = len(right_y) / len(y)
            score = w_left * gini_impurity(left_y) + w_right * gini_impurity(right_y)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_score


def build_tree(
    rows: List[List[float]],
    y: List[Any],
    depth: int,
    max_depth: int,
    min_samples_split: int,
    max_features: int,
) -> TreeNode:
    if len(set(y)) == 1:
        return TreeNode(value=y[0])
    if depth >= max_depth or len(rows) < min_samples_split:
        return TreeNode(value=majority_class(y))

    n_features = len(rows[0])
    feature_indices = random.sample(range(n_features), k=min(max_features, n_features))
    feature, threshold, _ = best_split(rows, y, feature_indices)

    if feature is None:
        return TreeNode(value=majority_class(y))

    left_rows, right_rows, left_y, right_y = split_rows(rows, y, feature, threshold)
    if not left_rows or not right_rows:
        return TreeNode(value=majority_class(y))

    left_node = build_tree(left_rows, left_y, depth + 1, max_depth, min_samples_split, max_features)
    right_node = build_tree(right_rows, right_y, depth + 1, max_depth, min_samples_split, max_features)

    return TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node)


def predict_tree(node: TreeNode, row: List[float]) -> Any:
    if node.value is not None:
        return node.value
    if row[node.feature] <= node.threshold:
        return predict_tree(node.left, row)
    return predict_tree(node.right, row)


def bootstrap_sample(rows: List[List[float]], y: List[Any]):
    n = len(rows)
    idx = [random.randrange(n) for _ in range(n)]
    return [rows[i] for i in idx], [y[i] for i in idx]


class RandomForestClassifier:
    def __init__(
        self,
        n_trees: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees: List[TreeNode] = []

    def fit(self, rows: List[List[float]], y: List[Any]) -> None:
        random.seed(self.seed)
        n_features = len(rows[0])
        max_features = self.max_features or max(1, int(n_features ** 0.5))

        self.trees = []
        for _ in range(self.n_trees):
            b_rows, b_y = bootstrap_sample(rows, y)
            tree = build_tree(
                b_rows,
                b_y,
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            self.trees.append(tree)

    def predict_one(self, row: List[float]) -> Any:
        votes = [predict_tree(tree, row) for tree in self.trees]
        return Counter(votes).most_common(1)[0][0]

    def predict(self, rows: List[List[float]]) -> List[Any]:
        return [self.predict_one(row) for row in rows]


def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


if __name__ == "__main__":
    X = [
        [2.7, 2.5], [1.3, 1.5], [3.0, 3.5], [2.0, 1.0],
        [7.0, 8.0], [8.5, 7.3], [9.1, 8.7], [7.7, 9.0],
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    clf = RandomForestClassifier(n_trees=30, max_depth=5, min_samples_split=2)
    clf.fit(X, y)
    pred = clf.predict(X)
    print("Pred:", pred)
    print("Accuracy:", round(accuracy(y, pred), 3))
