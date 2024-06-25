from base64 import b32encode

from loguru import logger as log
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
from blake3 import blake3
from platformdirs import PlatformDirs


APP_NAME = "iscc-sct"
APP_AUTHOR = "iscc"
dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
os.makedirs(dirs.user_data_dir, exist_ok=True)


__all__ = [
    "timer",
    "get_model",
    "encode_base32",
    "hamming_distance",
    "MODEL_PATH",
]


BASE_VERSION = "1.0.0"
BASE_URL = f"https://github.com/iscc/iscc-binaries/releases/download/v{BASE_VERSION}"
MODEL_FILENAME = "iscc-sct-v0.1.0.onnx"
MODEL_URL = f"{BASE_URL}/{MODEL_FILENAME}"
MODEL_PATH = Path(dirs.user_data_dir) / MODEL_FILENAME
MODEL_CHECKSUM = "ff254d62db55ed88a1451b323a66416f60838dd2f0338dba21bc3b8822459abc"


class timer:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        # Record the start time
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        # Calculate the elapsed time
        elapsed_time = time.perf_counter() - self.start_time
        # Log the message with the elapsed time
        log.debug(f"{self.message} {elapsed_time:.4f} seconds")


def get_model():  # pragma: no cover
    """Check and return local model file if it exists, otherwise download."""
    if MODEL_PATH.exists():
        try:
            return check_integrity(MODEL_PATH, MODEL_CHECKSUM)
        except RuntimeError:
            log.warning("Model file integrity error - redownloading ...")
            urlretrieve(MODEL_URL, filename=MODEL_PATH)
    else:
        log.info("Downloading embedding model ...")
        urlretrieve(MODEL_URL, filename=MODEL_PATH)
    return check_integrity(MODEL_PATH, MODEL_CHECKSUM)


def check_integrity(file_path, checksum):
    # type: (str|Path, str) -> Path
    """
    Check file integrity against blake3 checksum

    :param file_path: path to file to be checked
    :param checksum: blake3 checksum to verify integrity
    :raises RuntimeError: if verification fails
    """
    file_path = Path(file_path)
    file_hasher = blake3(max_threads=blake3.AUTO)
    with timer("INTEGRITY check time"):
        file_hasher.update_mmap(file_path)
        file_hash = file_hasher.hexdigest()
    if checksum != file_hash:
        msg = f"Failed integrity check for {file_path.name}"
        log.error(msg)
        raise RuntimeError(msg)
    return file_path


def encode_base32(data):
    # type: (bytes) -> str
    """
    Standard RFC4648 base32 encoding without padding.

    :param bytes data: Data for base32 encoding
    :return: Base32 encoded str
    """
    return b32encode(data).decode("ascii").rstrip("=")


def hamming_distance(a, b):
    # type: (bytes, bytes) -> int
    """
    Calculate the bitwise Hamming distance between two bytes objects.

    :param a: The first bytes object.
    :param b: The second bytes object.
    :return:  The Hamming distance between two bytes objects.
    :raise ValueError: If a and b are not the same length.
    """
    if len(a) != len(b):
        raise ValueError("The lengths of the two bytes objects must be the same")

    distance = 0
    for b1, b2 in zip(a, b):
        xor_result = b1 ^ b2
        distance += bin(xor_result).count("1")

    return distance
