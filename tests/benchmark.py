# -*- coding: utf-8 -*-
from loguru import logger as log
from pathlib import Path
import iscc_sct as sct
import argparse
import time


def benchmark(folder):
    """
    Benchmark Text-Code generation for all text files in `folder`.

    Per file stats are logged to the console during processing.
    Comprehensive aggregated statistics are shown after processing all images

    :param folder: Folder containing text files for benchmarking
    """
    folder = Path(folder)
    assert folder.is_dir(), f"{folder} is not a directory."

    total_time = 0
    file_count = 0

    for txt_path in folder.glob("*.txt"):
        start_time = time.time()
        try:
            iscc_meta = sct.code_text_semantic(txt_path)
        except Exception as e:
            log.error(f"Processing {txt_path.name} failed: {e}")
            continue
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        file_count += 1
        log.info(f"Processed {txt_path.name} in {elapsed_time:.2f} seconds. ISCC: {iscc_meta['iscc']}")

    if file_count > 0:
        avg_time = total_time / file_count
        log.info(
            f"Processed {file_count} files in {total_time:.2f} seconds. Average time per file: {avg_time:.2f} seconds."
        )
    else:
        log.warning("No text files found in the provided folder.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ISCC Semantic-Code Text generation.")
    parser.add_argument("folder", type=str, help="Directory containing text files for benchmarking.")
    args = parser.parse_args()

    benchmark(args.folder)


if __name__ == "__main__":
    main()
