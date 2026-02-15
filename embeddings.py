import re

if __name__ == "__main__":

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"No. of characters: {len(raw_text)}")
    print(f"Head: {raw_text[:99]}...")

    # Match punctuation, whitespace
    pttrn = r"[\W\s\n]+"

    # Split at above pattern
    # Remove last empty string: res.count("")
    res = re.split(pttrn, raw_text)[:-1]

    # Unnecessary
    # res = [x.strip() for x in res if re.match("\w+", x)]

    print(res)

    tokens = {idx: val for idx, val in enumerate(set(res))}

    print(tokens)
