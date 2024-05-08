"""Helper script do dump/freeze the current tokenizer"""

from tokenizers import Tokenizer
from pathlib import Path


HERE = Path(__file__).parent.absolute()


def main():
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save((HERE.parent / "iscc_sct/tokenizer.json").as_posix(), pretty=False)


if __name__ == "__main__":
    main()
