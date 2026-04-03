from itertools import islice
from random import shuffle

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
            logger.info(f"\t Node: {node}")
            logger.info(f"\t Children: {node._children}")
            logger.info(f"\t Local Grads: {node._local_grads}")
            for child, local_grad in zip(node._children, node._local_grads):
                logger.info(f"\t\t Child: {child}, Local Grad: {local_grad}")

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

    vocab = len(unique_chars) + 1

    a = Value(5.0)
    b = Value(2.0)
    c = Value(7.0)
    d = a * b * c

    d.backward()

    logger.info(a.grad)  # 14
    logger.info(b.grad)  # 35
    logger.info(c.grad)  # 10
