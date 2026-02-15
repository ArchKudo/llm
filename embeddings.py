import re


class WordTokens:

    def __init__(self, txt):

        res = self._process(txt)
        self.tokens = {val: idx for idx, val in enumerate(set(res))}
        self.decoded = {idx: val for val, idx in self.tokens.items()}

    def _process(self, txt):
        pttrn = re.compile(r"[\W\s\n]+")
        res = re.split(pttrn, txt)
        return res

    def encode(self, txt):
        res = self._process(txt)
        return [self.tokens[word] for word in res if word in self.tokens]

    def decode(self, tokens):
        return " ".join([self.decoded[token] for token in tokens])


if __name__ == "__main__":

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = f.read()

    embeddings = WordTokens(book)

    sent = embeddings.encode("silver trees are constantly destroyed")

    print(sent)

    print(embeddings.decode(sent))
