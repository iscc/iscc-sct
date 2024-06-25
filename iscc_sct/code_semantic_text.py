# -*- coding: utf-8 -*-
"""*A cross-lingual semantic similarity preserving hash for plain-text content (soft hash).*

The ISCC Text-Code Semantic is a content-based compact binary code generated from multilingual text.

!!! Warning

    This is a non-standard Proof of Concept implementation.
    Plain-text extraction from documents in various formats (especially PDF) may
    yield different results depending on the extraction tools being used.
    The [iscc-sdk](https://github.com/iscc/iscc-sdk) uses [Apache Tika](https://tika.apache.org)
    to extract text from documents for Text-Code generation.

**Algorithm overview**

- Split text into semantically coherent overlapping chunks.
- Create vector embeddings of the chunks.
- Average and binarize the chunk embeddings.
- Encode as ISCC-UNIT of MainType SEMANTIC and SubType TEXT
"""

from loguru import logger as log
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray
from functools import cache
import iscc_sct as sct


HERE = Path(__file__).parent.absolute()


__all__ = [
    "code_text_semantic",
    "gen_text_code_semantic",
    "soft_hash_text_semantic",
    "embed_chunks",
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


TOKENIZER_PATH = HERE / "tokenizer.json"
MAINTYPE = "0001"  # SEMANTIC
SUBTYPE = "0000"  # TEXT
SCT_VERSION = "0000"  # V0


def code_text_semantic(fp, **options):
    # type: (Path|str, Any) -> dict[str, Any]
    """
    Generate ISCC Semantic-Code Text from a text file.

    NOTE:
        If you enable generating granular features with `features=True` those features will have
        the same bit-length as the generated ISCC-UNIT.

    :param fp: File path of plaintext file to process
    :param options: Custom processing options for overriding global options
    :key bits (int): Length of generated Semantic Text-Code in bits (default 64)
    :key characters (bool): Return document character count (default True).
    :key embedding (bool): Return global document embedding (default False).
    :key precision (int): Max fractional digits for embeddings (default 8).
    :key features (bool): Return granular document features (default False).
    :key offsets (bool): Return character offsets for granular features (default False).
    :key chunks (bool): Return text chunks (default False).
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed to overlap between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: Dict with ISCC processing results
    """
    fp = Path(fp)
    return gen_text_code_semantic(fp.read_text(encoding="utf-8"), **options)


def gen_text_code_semantic(text, **options):
    # type: (str, Any) -> dict
    """
    Create an ISCC Semantic-Code Text from plaintext.

    :param str text: Plaint text for ISCC processing
    :param options: Custom processing options for overriding global options
    :key bits (int): Length of generated Semantic Text-Code in bits (default 64)
    :key characters (bool): Return document character count (default True).
    :key embedding (bool): Return global document embedding (default False).
    :key precision (int): Max fractional digits for embeddings (default 8).
    :key features (bool): Return granular document features (default False).
    :key offsets (bool): Return character offsets for granular features (default False).
    :key chunks (bool): Return text chunks (default False).
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed to overlap between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: Dict with ISCC processing results
    """

    if not text:
        raise ValueError("Input text cannot be empty.")

    opts = sct.sct_opts.override(options)

    result = {"iscc": None}  # Initialize first so `iscc` key is first in dict

    if opts.characters:
        result["characters"] = len(text)

    # Text splitting
    splits = split_text(text, **opts.model_dump())
    offsets, chunks = [list(item) for item in zip(*splits)]
    if opts.chunks:
        result["chunks"] = chunks
    if opts.offsets:
        result["offsets"] = offsets
    if opts.sizes:
        result["sizes"] = [len(chunk) for chunk in chunks]

    # Chunk embedding
    embeddings = embed_chunks(chunks)
    if opts.features:
        feature_digests = [binarize(vec)[: opts.bits // 8] for vec in embeddings]
        result["features"] = [sct.encode_base32(digest) for digest in feature_digests]

    # Create global document embedding
    embedding = mean_pooling(embeddings)
    if opts.embedding:
        result["embedding"] = compress(embedding, opts.precision)

    # Encode global document embedding
    length = BIT_LEN_MAP[opts.bits]
    header = int(MAINTYPE + SUBTYPE + SCT_VERSION + length, 2).to_bytes(2, byteorder="big")
    digest = binarize(embedding)[: opts.bits // 8]
    code = sct.encode_base32(header + digest)
    result["iscc"] = "ISCC:" + code
    return result


def soft_hash_text_semantic(text):
    # type: (str) -> bytes
    """Creates a 256-bit semantic similarity preserving hash for text input."""
    chunks = [item[1] for item in split_text(text)]
    embeddings = embed_chunks(chunks)
    embedding = mean_pooling(embeddings)
    digest = binarize(embedding)
    return digest


def split_text(text, **options):
    # type: (str) -> list[tuple[int,str]]
    """
    Split text into semantically coherent chunks for embedding.

    :param text: Text to split.
    :param options: Custom processing options for overriding global options
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed to overlap between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: A list of offset, chunk tuples [(offset,chunk), ...]
    """
    opts = sct.sct_opts.override(options)
    return splitter(**opts.model_dump()).chunk_indices(text)


@cache
def tokenizer():
    # type: () -> Tokenizer
    """
    Load and cache the tokenizer model based on the predefined model name.

    :return: An instance of the Tokenizer.
    """
    with sct.timer("TOKENIZER load time"):
        return Tokenizer.from_file(TOKENIZER_PATH.as_posix())


@cache
def splitter(**options):
    # type: (Any) -> TextSplitter
    """
    Load and cache the text splitter, initialized with tokenizer.

    :param options: Custom processing options for overriding global options
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed to overlap between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: An instance of TextSplitter.
    """
    opts = sct.sct_opts.override(options)
    with sct.timer("TEXTSPLITTER load time"):
        return TextSplitter.from_huggingface_tokenizer(
            tokenizer(), capacity=opts.max_tokens, overlap=opts.overlap, trim=opts.trim
        )


@cache
def model():
    # type: () -> rt.InferenceSession
    """
    Load and cache the ONNX inference model from a specified path.

    :return: An ONNX inference session.
    """
    with sct.timer("ONNXMODEL aquisition time"):
        model_path = sct.get_model()
    available_onnx_providers = rt.get_available_providers()
    log.debug(f"Available ONNX providers {', '.join(available_onnx_providers)}")
    selected_onnx_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_onnx_providers:  # pragma: no cover
        selected_onnx_providers.insert(0, "CUDAExecutionProvider")
    log.debug(f"Using ONNX providers {', '.join(selected_onnx_providers)}")
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    with sct.timer("ONNXMODEL load time"):
        # TODO assume model exists - and download onnx on failure - add environment info/check command
        return rt.InferenceSession(model_path, sess_options=so, providers=selected_onnx_providers)


def tokenize_chunks(chunks):
    # type: (list[str]) -> dict
    """
    Tokenize text chunks into model-compatible formats.

    :param chunks: Text chunks to tokenize.
    :return: Dictionary of tokenized data including input IDs, attention masks, and type IDs.
    """
    encodings = tokenizer().encode_batch(chunks)
    input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    type_ids = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": type_ids}


def embed_chunks(chunks):
    # type: (list[str]) -> NDArray[np.float32]
    """
    Embed text chunks and return vector embeddings.

    :param chunks: Text chunks to embed.
    :return: An array of embeddings for each chunk.
    """
    tokens = tokenize_chunks(chunks)
    token_embeddings = embed_tokens(tokens)
    return attention_pooling(token_embeddings, tokens["attention_mask"])


def embed_tokens(tokens):
    # type: (dict) -> NDArray
    """
    Create embeddings from tokenized text chunks using the model.

    :param tokens: Tokenized text data.
    :return: An array of embeddings.
    """
    result = model().run(None, tokens)
    return np.array(result[0])


def attention_pooling(token_embeddings, attention_mask):
    # type: (np.array, np.array) -> np.array
    """
    Apply attention mask based mean pooling to the token embeddings.

    :param token_embeddings: Raw token embeddings from the model.
    :param attention_mask: Attention masks for the embeddings.
    :return: An array of pooled and normalized embeddings.
    """
    input_mask_expanded = attention_mask[:, :, None].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    mean_pooled = sum_embeddings / sum_mask
    norm = np.linalg.norm(mean_pooled, ord=2, axis=1, keepdims=True)
    result = mean_pooled / np.clip(norm, a_min=1e-9, a_max=None)
    return result.astype(np.float32)


def mean_pooling(embeddings):
    # type: (NDArray[np.float32]) -> NDArray
    """
    Calculate the document vector from chunk embeddings using mean pooling.

    :param embeddings: Chunk embeddings.
    :return: A normalized document vector.
    """
    document_vector = embeddings.mean(axis=0)
    return document_vector / np.linalg.norm(document_vector)


def binarize(vec):
    # type: (NDArray) -> bytes
    """
    Binarize an embedding vector into a hash digest.

    :param vec: Vector to be binarized.
    :return: A bytes object representing the binary hash.
    """
    return bytes((np.packbits(np.array(vec) >= 0)))


def compress(vec, precision):
    # type: (NDArray, int) -> list[float]
    """
    Round down vector values to specified precision to reduce storage requirements.

    :param vec: Embedding vector.
    :param precision: Max number of fractional decimal places.
    :return: Vector as native python list of rounded floats.
    """
    rounded_array = np.around(vec, decimals=precision)
    compress_list = [round(x, precision) for x in rounded_array.tolist()]
    return compress_list
