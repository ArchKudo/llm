from itertools import islice
from random import shuffle

import re

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # 1 dataset
    with open("./The_Verdict.txt", "r", encoding="utf-8") as f:
        book = "".join(f.read().splitlines()[21:186])

    logger.debug(book[:20])

    sentences = set(sentence.strip() for sentence in book.split(".")) - {""}

    logger.debug(len(sentences))

    logger.debug(list(islice(sentences, 10)))

    words = re.findall(r"\w+", book)

    logger.info(words[:10])

    # 2 tokenizer
    # Let's build a word hallucinator first before building one for sentences

    shuffle(words)

    logger.info(words[:10])

    # Only encode characters in the text here a-zA-Z_
    unique_chars = sorted(set("".join(words)))

    logger.info(unique_chars)

    # Set Beginning of sequence as len + 1 for uniqueness
    BOS = len(unique_chars)

    vocab = len(unique_chars) + 1
