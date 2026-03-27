"""Educational scalar autograd engine inspired by reverse-mode AD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Set


@dataclass(eq=False)
class Value:
    data: float
    grad: float = 0.0
    _prev: Set["Value"] = field(default_factory=set)
    _backward: Callable[[], None] = lambda: None

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other: "Value") -> "Value":
        out = Value(self.data + other.data, _prev={self, other})

        def _bw() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _bw
        return out

    def __mul__(self, other: "Value") -> "Value":
        out = Value(self.data * other.data, _prev={self, other})

        def _bw() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _bw
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, _prev={self})

        def _bw() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _bw
        return out

    def backward(self) -> None:
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build(v: Value) -> None:
            if v in visited:
                return
            visited.add(v)
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    e = d.relu()
    e.backward()

    print("e:", e.data)
    print("grad a:", a.grad)
    print("grad b:", b.grad)
    print("grad c:", c.grad)


if __name__ == "__main__":
    main()
