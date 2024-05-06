from loguru import logger as log
from base64 import b32encode
from pathlib import Path
from typing import List, Tuple
import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray
import iscc_sct as sct


__all__ = [
    "code_text_semantic",
    "gen_text_code_semantic",
    "soft_hash_text_semantic",
    "preprocess_text",
]

BIT_LEN_MAP = {
    32: "0000",
    64: "0001",
    96: "0010",
    128: "0011",
    160: "0100",
    192: "0101",
    224: "0110",
    256: "0111",
}

# Lazy loaded ONNX model
_model = None


def code_text_semantic(text, bits=64):
    # type: (str, int) -> dict
    """
    Generate ISCC Semantic-Code Text from text input.

    :param str|Path text: Text for Semantic-Code creation.
    :param int bits: Bit-length of ISCC Semantic-Code Text (default 64, max 256).
    :return: ISCC metadata - `{"iscc": ..., "features": ...}`
    :rtype: dict
    """
    pass


def gen_text_code_semantic(arr, bits=64):
    # type: (NDArray[np.float32], int) -> dict
    """
    Create an ISCC Semantic-Code Image from normalized text embeddings.

    :param NDArray[np.float32] arr: Normalized text embeddings
    :param int bits: Bit-length of ISCC Semantic-Code Image (default 64, max 256).
    :return: ISCC Schema compatible dict with Semantic-Code Image.
    :rtype: dict
    """
    pass


def soft_hash_text_semantic(arr, bits=64):
    # type: (NDArray[np.float32], int) -> Tuple[bytes, NDArray[np.float32]]
    """
    Calculate semantic text hash from preprocessed text.

    :param NDArray[np.float32] arr: Preprocessed image array
    :param int bits: Bit-length of semantic image hash (default 64).
    :return: Tuple of image-hash digest and semantic feature vector from model.
    """


pass


def model():
    # type: () -> rt.InferenceSession
    """Initialize, cache and return inference model"""
    global _model
    if _model is None:
        model_path = sct.get_model()
        log.info(f"Initializing ONNX model for iscc-sci {sct.__version__}")
        with sct.metrics(name="ONNX load time {seconds:.2f} seconds"):
            _model = rt.InferenceSession(model_path)
    return _model


def preprocess_text(text):
    # type: (str) -> NDArray[np.float32]
    """Preprocess image for inference."""
    pass


def vectorize(arr):
    # type: (NDArray) -> List[NDArray[np.float32]]
    """Apply inference on tokenized text data"""
    pass


def binarize(vec):
    # type: (NDArray) -> bytes
    """Binarize vector embeddings."""

    bits = [1 if num >= 0 else 0 for num in vec]

    # Prepare a bytearray for the result
    result = bytearray()

    # Process each 8 bits (or the remaining in the last iteration)
    for i in range(0, len(bits), 8):
        # Convert 8 bits into a byte
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)
