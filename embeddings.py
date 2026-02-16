from importlib.metadata import version
import tiktoken

if __name__ == "__main__":

    print(f"""Verison: {version("tiktoken")}""")

    tokenizer = tiktoken.get_encoding("gpt2")

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = f.read()

    sent = tokenizer.encode(
        "silver trees are constantly destroyed, unacceptable!",
        allowed_special={"<|endoftext|>"},
    )

    print(sent)

    print(tokenizer.decode(sent))

    par = tokenizer.encode(
        """silver trees are destroyed.


        A new line for the unruly.
        I believe constantly even, unacceptable!""",
        allowed_special={"<|endoftext|>"},
    )

    print(par)

    print(tokenizer.decode(par))

    madeup = "sdffasdfasdf sdfasdfasdfsfasfsdfff asdfdfsfasdfasdfasfsfdasfsaf"

    print(tokenizer.encode(madeup))
    print(tokenizer.decode(tokenizer.encode(madeup)))
    print(tokenizer.decode([21282]))
