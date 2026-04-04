from itertools import islice
from random import shuffle, gauss

import math
import re

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Class to calculate gradient
class Value:

    # __slots__ is an attribute for restricting dynamic creation of fields
    # We apparently save memory by making them static
    # micro-optimizationgpt!
    # obj = Value(data)
    # obj.data = 10 (works)
    # obj.newfield = 20 (fails)
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):

        self.data = data
        self.grad = 0
        self._children = children
        # partial derivatives of output wrt each child
        # different from global grad which is calcuated on the final output after a long series of operations.
        # We calculate global grad of output
        # Store the local grad
        # Then for child we can apply chain rule to calculate global grad for child as local_grad * global_grad of output
        self._local_grads = local_grads

    def __repr__(self):

        data = f"Data: {self.data}"
        grad = f"Grad: {self.grad}"
        children = ",\n".join(repr(child) for child in self._children)
        local_grads = ",\n".join(repr(local_grad) for local_grad in self._local_grads)

        return (
            f"\nValue({data}, {grad},\n"
            f"\t_children=\n{children}\t_local_grads=\n{local_grads})"
        )

    def __add__(self, other):

        # Making it even tighter with even lesser handling
        # it is not done for __pow__ function anyway
        # But maybe it is because you always take pow with a scalar?
        # Resolved watching the micrograd video
        other = other if isinstance(other, Value) else Value(other)

        return Value(self.data + other.data, children=(self, other), local_grads=(1, 1))

    def __mul__(self, other):
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

    def __pow__(self, scalar):
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
        graph = []
        visited = set()

        # From video: topological sort of the graph
        def build_graph(node):
            logger.debug(f"Visiting node: {node}")
            if node not in visited:
                logger.debug(f"\tNode {node} not in {visited}")
                visited.add(node)
                for child in node._children:
                    logger.debug(
                        f"\t\tBuilding for child: {child} of {node._children} for {node}"
                    )
                    build_graph(child)
                graph.append(node)

        build_graph(self)

        logger.debug(f"Built graph for {self}:")
        for item in graph:
            logger.debug(f"\t{item}")

        logger.debug(f"Built rev graph for {self}:")
        for item in reversed(graph):
            logger.debug(f"\t{item}")

        self.grad = 1
        for node in reversed(graph):
            logger.debug(f"\t Node: {node}")
            logger.debug(f"\t Children: {node._children}")
            logger.debug(f"\t Local Grads: {node._local_grads}")
            for child, local_grad in zip(node._children, node._local_grads):
                logger.debug(f"\t\t Child: {child}, Local Grad: {local_grad}")

                # child.grad = d(output)/d(child) = d(node)/d(child) * d(output)/d(node) = local_grad * node.grad
                child.grad += local_grad * node.grad


if __name__ == "__main__":

    # 1 dataset
    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = "".join(f.read().splitlines()[21:186])

    logger.debug(book[:20])

    sentences = set(sentence.strip() for sentence in book.split(".")) - {""}

    logger.debug(len(sentences))

    logger.debug(list(islice(sentences, 10)))

    words = re.findall(r"\w+", book)

    logger.debug(words[:10])

    # 2 tokenizer
    # Let's build a word hallucinator first before building one for sentences

    shuffle(words)

    logger.debug(words[:10])

    # Only encode characters in the text here a-zA-Z_
    unique_chars = sorted(set("".join(words)))

    logger.debug(unique_chars)

    # Set Beginning of sequence as len + 1 for uniqueness
    BOS = len(unique_chars)

    vocab_sz = len(unique_chars) + 1

    a = Value(5.0)
    b = Value(2.0)
    c = Value(7.0)
    d = a * b * c

    d.backward()

    logger.debug(a.grad)  # 14
    logger.debug(b.grad)  # 35
    logger.debug(c.grad)  # 10

    # 4 Parameters

    n_embed = 16
    n_head = 4
    n_layer = 1
    block_sz = 16

    head_dim = n_embed // n_head

    def matrix(nout, nin, std=0.08):
        return [[Value(gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {
        "wte": matrix(vocab_sz, n_embed),
        "wpe": matrix(block_sz, n_embed),
        "lm_head": matrix(vocab_sz, n_embed),
    }

    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embed, n_embed)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embed, n_embed)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embed, n_embed)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embed, n_embed)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embed, n_embed)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embed, 4 * n_embed)

    # flatten state dict
    params = [p for mat in state_dict.values() for row in mat for p in row]

    logger.info(len(params))

    for k, v in state_dict.items():
        logger.info(f"Table: {k} - {len(v)} x {len(v[0])}")

    # 5 Architecture

    # matrix multiply vector with weights
    # y = Wv
    def linear(vec, weight):
        return [sum(wi * vi for wi, vi in zip(wo, vec)) for wo in weight]

    # convert from domain -inf - inf to 0,1
    # e^ to make it positive and exp change for small values
    # divide by sum to normalize
    # softmax(zi) = exp(zi - m) / Σj=1->n exp(zj - m)
    def softmax(logits):
        maxval = max(val.data for val in logits)
        exps = [(val - maxval).exp() for val in logits]
        total = sum(exps)
        return [exp / total for exp in exps]

    # rms = (1/n * Σxi^2 + eps)^-1/2
    # eps for keeping denominator always non-zero
    def rmsnorm(x, eps=1e-5):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + eps) ** -0.5

        return [xi * scale for xi in x]

    def gpt(tokid, posid, keys, vals):

        tokemb = state_dict["wte"][tokid]
        posemb = state_dict["wpe"][posid]

        x = [t + p for t, p in zip(tokemb, posemb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            xresidual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])

            keys[li].append(k)
            values[li].append(v)

            xattn = []

            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs : hs + head_dim]
                k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + head_dim] for vi in values[li]]

                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(len(k_h))
                ]

                attn_weights = softmax(attn_logits)

                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]

                xattn.extend(head_out)

            x = linear(xattn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, xresidual)]

            xresidual = x
            x = rmsnorm(x)

            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])

            x = [a + b for a, b in zip(x, xresidual)]

        logits = linear(x, state_dict["lm_head"])

        return logits

    # 6 Training loop

    λ, β1, β2, ε = 0.01, 0.85, 0.99, 1e-8

    m = [0.0] * len(params)
    v = [0.0] * len(params)

    n_steps = 1000
    for step in range(n_steps):
        word = words[step % len(words)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in word] + [BOS]
        n = min(block_sz, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

        losses = []

        for posid in range(n):
            tokenid, targetid = tokens[posid], tokens[posid + 1]
            logits = gpt(tokenid, posid, keys, values)

            probs = softmax(logits)

            losst = -probs[targetid].log()
            losses.append(losst)

        loss = (1 / n) * sum(losses)

        loss.backward()

        lrt = λ * (1 - step / n_steps)

        for i, param in enumerate(params):
            m[i] = β1 * m[i] + (1 - β1) * param.grad
            v[i] = β2 * v[i] + (1 - β2) * param.grad**2

            m_hat = m[i] / (1 - β1 ** (step + 1))
            v_hat = v[i] / (1 - β2 ** (step + 1))

            param.data -= lrt * m_hat / (v_hat**0.5 + ε)
            param.grad = 0

        logger.info(f"Steps: {step + 1:4d} / {n_steps:4d} |" f"Loss: {loss.data:.4f}")
