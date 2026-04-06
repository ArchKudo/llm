from __future__ import annotations

from random import shuffle, gauss, choices
from typing import List, Tuple, Union

import math
import re

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# 4 Parameters

N_EMBED: int = 16
N_HEAD: int = 4
N_LAYER: int = 1
BLOCK_SZ: int = 16

HEAD_DIM: int = N_EMBED // N_HEAD  # 2


# Class to calculate gradient
class Value:
    # __slots__ is an attribute for restricting dynamic creation of fields
    # We apparently save memory by making them static
    # micro-optimizationgpt!
    # obj = Value(data)
    # obj.data = 10 (works)
    # obj.newfield = 20 (fails)
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(
        self,
        data: float,
        children: Tuple["Value", ...] = (),
        local_grads: Tuple[float, ...] = (),
    ):

        self.data: float = data
        self.grad: float = 0
        self._children: Tuple[Value, ...] = children
        # partial derivatives of output wrt each child
        # different from global grad which is calcuated on the final output after a long series of operations.
        # We calculate global grad of output
        # Store the local grad
        # Then for child we can apply chain rule to calculate global grad for child as local_grad * global_grad of output
        self._local_grads: Tuple[float, ...] = local_grads

    def __repr__(self):

        # data: str = f"Data: {self.data}"
        # grad: str = f"Grad: {self.grad}"
        # children: str = ",\n".join(repr(child) for child in self._children)
        # local_grads: str = ",\n".join(
        #     repr(local_grad) for local_grad in self._local_grads
        # )

        return f"Value(data={self.data}, grad={self.grad}, _children={self._children}, _local_grads={self._local_grads})"

    def __add__(self, other: Union[Value, float]):

        # Making it even tighter with even lesser handling
        # it is not done for __pow__ function anyway
        # But maybe it is because you always take pow with a scalar?
        # Resolved watching the micrograd video
        other = other if isinstance(other, Value) else Value(other)

        return Value(self.data + other.data, children=(self, other), local_grads=(1, 1))

    def __mul__(self, other: Union[Value, float]):
        other = other if isinstance(other, Value) else Value(other)

        # So that i don't have to jog memory each time
        # other.data, self.data because we are taking partial derivatives first wrt x, x'
        # making that value 1 but keeps coefficient as is
        # other.data is result of d(self.data, other.data)/d(self.data)
        return Value(
            self.data * other.data,
            children=(self, other),
            local_grads=(other.data, self.data),
        )

    def __pow__(self, scalar: float):
        # Another mem jog trailing comma to treat it as tuple even with single value`
        return Value(
            self.data**scalar,
            children=(self,),
            local_grads=(scalar * self.data ** (scalar - 1),),
        )

    def log(self):
        return Value(
            math.log(self.data), children=(self,), local_grads=(1 / self.data,)
        )

    def exp(self):
        return Value(
            math.exp(self.data), children=(self,), local_grads=(math.exp(self.data),)
        )

    def relu(self):
        return Value(
            max(0, self.data), children=(self,), local_grads=(float(self.data > 0),)
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        """
        __radd__ is for right addition when the left operator doesn't support addition
        i.e.
        5 + Value()
        runs Value().__add__(5)
        """
        return self + other

    def __sub__(self, other):

        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * (self**-1)

    def backward(self):
        graph: list[Value] = []
        visited: set[Value] = set()

        # From video: topological sort of the graph
        # def build_graph(node: Value):
        #     logger.debug(f"Visiting node: {node}")
        #     if node not in visited:
        #         logger.debug(f"\tNode {node} not in {visited}")
        #         visited.add(node)

        #         for child in node._children:
        #             logger.debug(
        #                 f"\t\tBuilding for child: {child} of {node._children} for {node}"
        #             )
        #             build_graph(child)
        #         graph.append(node)

        def build_graph(node: Value):
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(
                        child for child in curr._children if child not in visited
                    )
                    graph.append(curr)

        build_graph(self)

        self.grad = 1
        for node in reversed(graph):
            for child, local_grad in zip(node._children, node._local_grads):
                # child.grad = d(output)/d(child) = d(node)/d(child) * d(output)/d(node) = local_grad * node.grad
                child.grad += local_grad * node.grad


def load_dataset(path: str = "./The_Verdict.txt", slice=slice(21, 186)) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        book: str = "".join(f.read().splitlines()[slice])

    # sentences: set[str] = set(sentence.strip() for sentence in book.split(".")) - {""}

    words: list[str] = re.findall(r"\w+", book)
    shuffle(words)

    return words


def tokenize(words: list[str]) -> Tuple[list[str], int, int]:
    charset: list[str] = sorted(set("".join(words)))
    BOS: int = len(charset)
    vocab_sz: int = len(charset) + 1

    return charset, BOS, vocab_sz


def matrix(nout: int, nin: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def state(vocab_sz: int) -> Tuple[dict[str, list[list[Value]]], list[Value]]:
    state_dict: dict[str, list[list[Value]]] = {
        # word token embedding
        "wte": matrix(vocab_sz, N_EMBED),
        # word position embedding
        "wpe": matrix(BLOCK_SZ, N_EMBED),
        # Language model head
        "lm_head": matrix(vocab_sz, N_EMBED),
    }

    for i in range(N_LAYER):
        # Attention query, key, value, output weights
        state_dict[f"layer{i}.attn_wq"] = matrix(N_EMBED, N_EMBED)
        state_dict[f"layer{i}.attn_wk"] = matrix(N_EMBED, N_EMBED)
        state_dict[f"layer{i}.attn_wv"] = matrix(N_EMBED, N_EMBED)
        state_dict[f"layer{i}.attn_wo"] = matrix(N_EMBED, N_EMBED)
        # First and second feed-forward layer weights
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * N_EMBED, N_EMBED)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(N_EMBED, 4 * N_EMBED)

    # Flatten state_dict into a single list of Value objects for optimization
    state_params: list[Value] = [
        elem for mat in state_dict.values() for row in mat for elem in row
    ]

    return state_dict, state_params


def linear(vec: list[Value], weight: list[list[Value]]) -> list[Value]:
    """
    # matrix multiply vector with weights
    # y = Wv
    """
    return [sum(wi * vi for wi, vi in zip(wo, vec)) for wo in weight]


def softmax(logits: list[Value]) -> list[Value]:
    """
    # convert from domain -inf - inf to 0,1
    # e^ to make it positive and exp change for small values
    # divide by sum to normalize
    # softmax(zi) = exp(zi - m) / Σj=1->n exp(zj - m)
    """
    maxval = max(val.data for val in logits)
    exps = [(val - maxval).exp() for val in logits]
    total = sum(exps)
    return [exp / total for exp in exps]


def rmsnorm(x: list[Value], eps: float = 1e-5) -> list[Value]:
    """
    # RMS = (1/n * Σxi^2 + eps)^-1/2
    # eps for keeping denominator always non-zero
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + eps) ** -0.5

    return [xi * scale for xi in x]


def gpt(
    tokid: int,
    posid: int,
    keys: list[list[list[Value]]],
    vals: list[list[list[Value]]],
    state_dict: dict[str, list[list[Value]]],
) -> list[Value]:

    tokemb: list[Value] = state_dict["wte"][tokid]
    posemb: list[Value] = state_dict["wpe"][posid]

    x: list[Value] = [t + p for t, p in zip(tokemb, posemb)]
    logger.info(f"max x: {max(x, key=lambda xi: xi.data).data}")

    x = rmsnorm(x)
    logger.info(f"max x after rmsnorm: {max(x, key=lambda xi: xi.data).data}")

    for li in range(N_LAYER):
        logger.info(f"Layer {li + 1} / {N_LAYER}")

        xresidual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        keys[li].append(k)
        vals[li].append(v)

        xattn: list[Value] = []

        for h in range(N_HEAD):
            logger.info(f"Layer {li + 1} / {N_LAYER} - Head {h + 1} / {N_HEAD}")

            hs = h * HEAD_DIM
            q_h = q[hs : hs + HEAD_DIM]
            k_h = [ki[hs : hs + HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs : hs + HEAD_DIM] for vi in vals[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM**0.5
                for t in range(len(k_h))
            ]

            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]

            xattn.extend(head_out)

        x = linear(xattn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, xresidual)]

        xresidual = x
        x = rmsnorm(x)

        # MLP
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, xresidual)]

    logits = linear(x, state_dict["lm_head"])

    return logits


if __name__ == "__main__":
    # 1 dataset
    words = load_dataset()
    # words = ["apple", "banana", "mango", "chikoo", "orange", "pineapple"]

    # 2 tokenizer
    # Let's build a word hallucinator first before building one for sentences
    charset, BOS, vocab_sz = tokenize(words)

    # 4 state variables
    state_dict, state_params = state(vocab_sz)
    logger.debug(len(state_params))
    for k, v in state_dict.items():
        logger.debug(f"Table: {k} - {len(v)} x {len(v[0])}")

    # 5 Architecture

    # 6 Training loop

    λ, β1, β2, ε = 0.01, 0.85, 0.99, 1e-8

    logger.info(f"λ: {λ}, β1: {β1}, β2: {β2}, ε: {ε}")

    m = [0.0] * len(state_params)
    v = [0.0] * len(state_params)

    n_steps = 1000
    for step in range(n_steps):
        logger.info(f"Training step {step + 1} / {n_steps}")
        word = words[step % len(words)]
        tokens = [BOS] + [charset.index(ch) for ch in word] + [BOS]
        n = min(BLOCK_SZ, len(tokens) - 1)

        keys, values = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]

        losses = []

        for posid in range(n):
            tokenid, targetid = tokens[posid], tokens[posid + 1]
            logits = gpt(tokenid, posid, keys, values, state_dict)

            probs = softmax(logits)

            losst = -probs[targetid].log()
            losses.append(losst)

        loss = (1 / n) * sum(losses)

        loss.backward()

        lrt = λ * (1 - step / n_steps)

        for i, param in enumerate(state_params):
            logger.info(f"Updating parameter {i + 1} / {len(state_params)}")
            m[i] = β1 * m[i] + (1 - β1) * param.grad
            v[i] = β2 * v[i] + (1 - β2) * param.grad**2

            m_hat = m[i] / (1 - β1 ** (step + 1))
            v_hat = v[i] / (1 - β2 ** (step + 1))

            param.data -= lrt * m_hat / (v_hat**0.5 + ε)
            param.grad = 0

        logger.info(f"Steps: {step + 1:4d} / {n_steps:4d} |Loss: {loss.data:.4f}")

    # 7 Inference
    temperature = 0.5

    for idx in range(20):
        k, v = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]

        tokid = BOS
        sample = []

        for posid in range(BLOCK_SZ):
            logits = gpt(tokid, posid, k, v, state_dict)
            probs = softmax([(logit / temperature) for logit in logits])
            tokid = choices(range(vocab_sz), weights=[p.data for p in probs])[0]
            if tokid == BOS:
                break
            sample.append(charset[tokid])

        print(f"sample: {idx + 1:2d} - {''.join(sample)}")
