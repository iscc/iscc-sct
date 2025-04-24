import math
from base64 import b32encode, b32decode
from pybase64 import urlsafe_b64encode, urlsafe_b64decode
from loguru import logger as log
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
from blake3 import blake3
from platformdirs import PlatformDirs
from typing import List, Tuple
from iscc_sct.models import Metadata, Feature


APP_NAME = "iscc-sct"
APP_AUTHOR = "iscc"
dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
os.makedirs(dirs.user_data_dir, exist_ok=True)


__all__ = [
    "timer",
    "get_model",
    "encode_base32",
    "encode_base64",
    "decode_base32",
    "decode_base64",
    "hamming_distance",
    "iscc_distance",
    "cosine_similarity",
    "granular_similarity",
    "char_to_byte_offsets",
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


def decode_base32(code):
    # type: (str) -> bytes
    """
    Standard RFC4648 base32 decoding without padding and with casefolding.
    """
    # python stdlib does not support base32 without padding, so we have to re-pad.
    cl = len(code)
    pad_length = math.ceil(cl / 8) * 8 - cl

    return bytes(b32decode(code + "=" * pad_length, casefold=True))


def encode_base64(data):
    # type: (bytes) -> str
    """
    Standard RFC4648 base64url encoding without padding.
    """
    code = urlsafe_b64encode(data).decode("ascii")
    return code.rstrip("=")


def decode_base64(code):
    # type: (str) -> bytes
    """
    Standard RFC4648 base64url decoding without padding.
    """
    padding = 4 - (len(code) % 4)
    string = code + ("=" * padding)
    return urlsafe_b64decode(string)


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


def iscc_distance(iscc1, iscc2):
    # type: (str, str) -> int
    """
    Calculate the Hamming distance between two ISCC Semantic Text Codes.

    :param iscc1: The first ISCC Semantic Text Code.
    :param iscc2: The second ISCC Semantic Text Code.
    :return: The Hamming distance between the two ISCC codes.
    :raise ValueError: If the input ISCCs are not valid or of different lengths.
    """
    # Remove the "ISCC:" prefix if present
    iscc1 = iscc1[5:] if iscc1.startswith("ISCC:") else iscc1
    iscc2 = iscc2[5:] if iscc2.startswith("ISCC:") else iscc2

    # Decode the base32-encoded ISCCs
    decoded1 = decode_base32(iscc1)
    decoded2 = decode_base32(iscc2)

    # Check if the decoded ISCCs have the same length
    if len(decoded1) != len(decoded2):
        raise ValueError("The input ISCCs must have the same length")

    # Remove the 2-byte header from each decoded ISCC
    content1 = decoded1[2:]
    content2 = decoded2[2:]

    # Calculate and return the Hamming distance
    return hamming_distance(content1, content2)


def cosine_similarity(a, b):
    # type: (bytes, bytes) -> int
    """
    Calculate the approximate cosine similarity based on Hamming distance for two bytes inputs.

    :param a: The first bytes object.
    :param b: The second bytes object.
    :return: The approximate cosine similarity between the two inputs, scaled from -100 to +100.
    :raise ValueError: If a and b are not the same length.
    """
    if len(a) != len(b):
        raise ValueError("The lengths of the two bytes objects must be the same")

    distance = hamming_distance(a, b)
    total_bits = len(a) * 8
    similarity = 1 - (2 * distance / total_bits)
    return max(min(int(similarity * 100), 100), -100)


def granular_similarity(metadata_a, metadata_b, threshold=80):
    # type: (Metadata, Metadata, int) -> List[Tuple[Feature, int, Feature]]
    """
    Compare simprints from two Metadata objects and return matching pairs above a similarity
    threshold. Only the most similar pair for each simprint_a is included.

    :param metadata_a: The first Metadata object.
    :param metadata_b: The second Metadata object.
    :param threshold: The similarity threshold (0-100) above which simprints are considered a match.
    :return: A list of tuples containing matching simprints and their similarity.
    """
    metadata_a = metadata_a.to_object_format()
    metadata_b = metadata_b.to_object_format()

    matches = []

    for feature_set_a in metadata_a.features:
        for simprint_a in feature_set_a.simprints:
            best_match = None
            best_similarity = threshold - 1

            for feature_set_b in metadata_b.features:
                for simprint_b in feature_set_b.simprints:
                    similarity = cosine_similarity(
                        decode_base64(simprint_a.simprint), decode_base64(simprint_b.simprint)
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (simprint_a, similarity, simprint_b)

            if best_match:
                matches.append(best_match)

    return matches


def char_to_byte_offsets(text, char_positions):
    # type: (str, list[int]) -> list[int]
    """
    Efficiently convert character positions to byte positions in a single pass.

    :param text: The input text
    :param char_positions: List of character positions to convert
    :return: List of corresponding byte positions
    """
    if not char_positions:
        return []

    # Sort positions for efficient single-pass processing
    sorted_positions = sorted(set(char_positions))
    pos_map = {pos: idx for idx, pos in enumerate(sorted_positions)}
    byte_positions = [0] * len(sorted_positions)

    char_pos = byte_pos = 0
    pos_idx = 0

    for ch in text:
        if pos_idx < len(sorted_positions) and char_pos == sorted_positions[pos_idx]:
            byte_positions[pos_idx] = byte_pos
            pos_idx += 1

        # Efficient branch-free UTF-8 byte length calculation
        cp = ord(ch)
        byte_pos += 1 + (cp >= 0x80) + (cp >= 0x800) + (cp >= 0x10000)
        char_pos += 1

    # After processing all characters, handle any requested position equal to len(text)
    while pos_idx < len(sorted_positions) and sorted_positions[pos_idx] == char_pos:
        byte_positions[pos_idx] = byte_pos
        pos_idx += 1

    # Map back to original order
    return [byte_positions[pos_map[pos]] for pos in char_positions]


def char_to_byte_offsets_simple(text, char_positions):
    # type: (str, list[int]) -> list[int]
    """
    Simple implementation to convert character positions to byte offsets in a UTF-8 encoded string.
    This function repeatedly encodes text slices, so its performance is not optimal for large texts.

    :param text: The input text.
    :param char_positions: List of character positions.
    :return: List of corresponding byte positions.
    """
    return [len(text[:pos].encode("utf-8")) for pos in char_positions]
