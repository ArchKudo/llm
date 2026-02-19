from importlib.metadata import version
import tiktoken

if __name__ == "__main__":

    print(f"""Verison: {version("tiktoken")}""")

    tokenizer = tiktoken.get_encoding("gpt2")

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = f.read()

    encoded = tokenizer.encode(book)

    print(len(encoded))

    context_sz = 4

    x = encoded[:context_sz]
    y = encoded[1 : context_sz + 1]

    print(x)
    print(f"\t{y}")

    nextword = [
        (encoded[i : context_sz + i], encoded[context_sz + i])
        for i in range(len(encoded) // context_sz)
    ]

    print(encoded[-80:])
    print(nextword[-5:])
