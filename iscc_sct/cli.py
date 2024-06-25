import argparse
import glob
from pathlib import Path
from loguru import logger
from iscc_sct.main import create
from charset_normalizer import from_bytes


def main():
    parser = argparse.ArgumentParser(description="Generate Semantic Text-Codes for text files.")
    parser.add_argument("path", type=str, help="Path to text files (supports glob patterns).", nargs="?")
    parser.add_argument("-b", "--bits", type=int, default=256, help="Bit-Length of Code (default 256)")
    parser.add_argument("-g", "--granular", action="store_true", help="Activate granular processing.")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debugging messages.")
    args = parser.parse_args()

    if args.path is None:
        parser.print_help()
        return

    if not args.debug:
        logger.remove()

    for path in glob.glob(args.path):
        path = Path(path)
        if path.is_file():
            logger.debug(f"Processing {path.name}")
            with path.open("rb") as file:
                data = file.read()
                try:
                    text = data.decode("utf-8")
                    if not text.strip():
                        logger.warning(f"SKIPPED empty: {path}")
                        continue
                except UnicodeDecodeError:
                    logger.debug(f"Could not decode {path.name} as UTF-8.")
                    charset_match = from_bytes(data).best()
                    if not charset_match:  # pragma: no cover
                        logger.error(f"SKIPPING {path.name} - failed to detect text encoding")
                        continue
                    logger.debug(f"Decode {path.name} with {charset_match.encoding}.")
                    text = str(charset_match)
                sct_meta = create(text, granular=args.granular, bits=args.bits)
                if args.granular:
                    print(repr(sct_meta))
                else:
                    print(sct_meta.iscc)


if __name__ == "__main__":  # pragma: no cover
    main()
