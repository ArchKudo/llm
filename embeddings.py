import re


class WordTokens:

    def __init__(self, txt):

        res = self._process(txt)
        res += ["<unk>", "<end>"]

        self.tokens = {val: idx for idx, val in enumerate(set(res))}

        self.decoded = {idx: val for val, idx in self.tokens.items()}

    def _process(self, txt):
        pttrn = re.compile(r"[\W\s\n]+")
        res = re.split(pttrn, txt)
        return res

    def _encode(self, txt):
        res = self._process(txt)

        unk = self.tokens["<unk>"]
        print(unk)
        return [self.tokens.get(word, unk) for word in res]

    def decode(self, tokens):
        return " ".join([self.decoded[token] for token in tokens])

    def encode(self, *txts):
        end = self.tokens["<end>"]

        res = []

        for txt in txts:
            res += self._encode(txt)
            res.append(end)

        return res


if __name__ == "__main__":

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = f.read()

    embeddings = WordTokens(book)

    sent = embeddings.encode("silver trees are constantly destroyed, unacceptable!")

    print(sent)

    print(embeddings.decode(sent))

    par = embeddings.encode(
        "silver trees are destroyed", "I believe constantly even, unacceptable"
    )

    print(par)

    print(embeddings.decode(par))
