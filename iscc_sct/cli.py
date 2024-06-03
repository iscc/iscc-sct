import argparse
import glob
import os
from loguru import logger
from iscc_sct.main import create


def main():
    parser = argparse.ArgumentParser(description="Generate Semantic Text-Codes for text files.")
    parser.add_argument("path", type=str, help="Path to text files (supports glob patterns).")
    parser.add_argument("--granular", action="store_true", help="Activate granular processing.")
    parser.add_argument("--debug", action="store_true", help="Show debugging messages.")
    args = parser.parse_args()
    if not args.debug:
        logger.remove()

    for filepath in glob.glob(args.path):
        if os.path.isfile(filepath):
            with open(filepath, "rt", encoding="utf-8") as file:
                text = file.read()
                sct_meta = create(text, granular=args.granular)
                if args.granular:
                    print(sct_meta.model_dump_json(indent=2, exclude_none=True))
                else:
                    print(sct_meta.iscc)


if __name__ == "__main__":
    main()
