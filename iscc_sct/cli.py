# -*- coding: utf-8 -*-
import argparse
import glob
import os
from iscc_sct.main import create


def main():
    parser = argparse.ArgumentParser(description="Generate Semantic Text-Codes for text files.")
    parser.add_argument("path", type=str, help="Path to text files (supports glob patterns).")
    parser.add_argument("--granular", action="store_true", help="Activate granular processing.")
    args = parser.parse_args()

    for filepath in glob.glob(args.path):
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                sct_meta = create(text, granular=args.granular)
                print(f"File: {filepath}\nSemantic Text-Code: {sct_meta.iscc}\n")


if __name__ == "__main__":
    main()
