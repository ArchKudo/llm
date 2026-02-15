
if __name__ == "__main__":

    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"No. of characters: {len(raw_text)}")
    print(f"Head: {raw_text[:99]}...")
