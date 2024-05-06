from loguru import logger as log
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
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


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Lazy loaded model, tokenizer, splitter
_model = None
_tokenizer = None
_splitter = None


def code_text_semantic(fp, bits=64):
    # type: (str|Path, int) -> dict
    """
    Generate ISCC Semantic-Code Text from text input.

    :param str|Path fp: Text filepath used for Semantic-Code creation.
    :param int bits: Bit-length of ISCC Semantic-Code Text (default 64, max 256).
    :return: ISCC metadata - `{"iscc": ..., "features": ...}`
    :rtype: dict
    """
    return gen_text_code_semantic(fp.read_text(encoding="utf-8"))


def gen_text_code_semantic(text, bits=64):
    # type: (str, int) -> dict
    """
    Create an ISCC Semantic-Code Text from plaintext.

    :param str text: Normalized text embeddings
    :param int bits: Bit-length of ISCC Semantic-Code Text (default 64, max 256).
    :return: ISCC Schema compatible dict with Semantic-Code Text.
    :rtype: dict
    """
    if bits < 32 or bits % 32:
        raise ValueError(f"Invalid bitlength {bits}")

    mtype = "0001"  # SEMANTIC
    stype = "0000"  # TEXT
    version = "0000"  # V0
    length = BIT_LEN_MAP[bits]
    header = int(mtype + stype + version + length, 2).to_bytes(2, byteorder="big")

    features = embed_text(text)
    digest = binarize(features)
    digest = digest[: bits // 8]
    code = b32encode(header + digest).decode("ascii").rstrip("=")
    iscc = "ISCC:" + code
    return {"iscc": iscc, "features": features.tolist()}


def soft_hash_text_semantic(arr, bits=64):
    # type: (NDArray[np.float32], int) -> Tuple[bytes, NDArray[np.float32]]
    """
    Calculate semantic text hash from preprocessed text.

    :param NDArray[np.float32] arr: Preprocessed image array
    :param int bits: Bit-length of semantic image hash (default 64).
    :return: Tuple of image-hash digest and semantic feature vector from model.
    """
    pass


def splitter():
    # type: () -> TextSplitter
    """Initialize, cache and return splitter"""
    global _splitter
    if _splitter is None:
        log.debug(f"Initializing splitter for iscc-sct {sct.__version__}")
        with sct.metrics(name="ONNX load time {seconds:.2f} seconds"):
            _splitter = TextSplitter.from_huggingface_tokenizer(
                tokenizer(),
                capacity=127,
                overlap=48,
                trim=False,
            )
    return _splitter


def tokenizer():
    # type: () -> Tokenizer
    """Initialize, cache and return tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        log.debug(f"Initializing tokenizer for iscc-sct {sct.__version__}")
        with sct.metrics(name="ONNX load time {seconds:.2f} seconds"):
            _tokenizer = Tokenizer.from_pretrained(model_name)
    return _tokenizer


def model():
    # type: () -> rt.InferenceSession
    """Initialize, cache and return inference model"""
    global _model
    if _model is None:
        model_path = sct.get_model()
        log.debug(f"Initializing ONNX model for iscc-sct {sct.__version__}")
        with sct.metrics(name="ONNX load time {seconds:.2f} seconds"):
            _model = rt.InferenceSession(model_path)
    return _model


def split_text(text):
    # type: (str) -> List[str]
    """Split text into chunks for embedding"""
    return splitter().chunks(text)


def tokenize_chunks(chunks):
    # type: (List[str]) -> dict
    """Tokenize text chunks"""
    encodings = tokenizer().encode_batch(chunks)
    input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    type_ids = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": type_ids}


def embed_text(text):
    # type: (str) -> NDArray
    """Create global text embedding"""
    chunks = split_text(text)
    chunks_embeddings = embed_chunks(chunks)
    text_embedding = mean_pooling(chunks_embeddings)
    return text_embedding


def embed_chunks(chunks):
    # type: (List[str]) -> NDArray[np.float32]
    """Embed text chunks"""
    tokens = tokenize_chunks(chunks)
    token_embeddings = embed_tokens(tokens)
    return attention_pooling(token_embeddings, tokens["attention_mask"])


def embed_tokens(tokens):
    # type: (dict) -> np.array
    """Create embeddigns from tokenized text chunks"""
    result = model().run(None, tokens)
    return np.array(result[0])


def attention_pooling(token_embeddings, attention_mask):
    # type: (np.array, np.array) -> np.array
    """Attention mask based mean pooling of inference results"""
    input_mask_expanded = attention_mask[:, :, None].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    mean_pooled = sum_embeddings / sum_mask
    norm = np.linalg.norm(mean_pooled, ord=2, axis=1, keepdims=True)
    result = mean_pooled / np.clip(norm, a_min=1e-9, a_max=None)
    return result.astype(np.float32)


def mean_pooling(embeddings):
    # type: (NDArray[np.float32]) -> NDArray
    """Calculate document vector form chunk embeddings"""
    document_vector = embeddings.mean(axis=0)
    return document_vector / np.linalg.norm(document_vector)


def preprocess_text(text):
    # type: (str) -> NDArray[np.float32]
    """Tokenizes text for inference."""
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
