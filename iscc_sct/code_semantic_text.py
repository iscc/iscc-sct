# -*- coding: utf-8 -*-
"""*A semantic similarity preserving hash for plain-text content (soft hash).*

The ISCC Text-Code Semantic is generated from plain-text that has been extracted from a media asset.

!!! Warning

    Plain-text extraction from documents in various formats (especially PDF) may
    yield diffent results depending on the extraction tools being used.
    The [iscc-sdk](https://github.com/iscc/iscc-sdk) uses [Apache Tika](https://tika.apache.org)
    to extract text from documents for Text-Code generation.

**Algorithm overview**

- Split text into semantically coherent overlapping chunks.
- Create vector embeddings of the chunks.
- Average and binarize the chunk embeddings.
- Count characters of collapsed text
- Apply [`soft_hash_text_v0`][iscc_core.code_content_text.soft_hash_text_v0] to collapsed text
"""

from loguru import logger as log
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray
from functools import cache
import iscc_sct as sct


HERE = Path(__file__).parent.absolute()


__all__ = [
    "code_text_semantic",
    "gen_text_code_semantic",
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
CAPACITY = 127  # Maximum number of tokens per chunk
OVERLAP = 48  # Maximum number of allowed tokens to overlap between chunks
TRIM = False  # Weather to trim whitespace on token chunks


def code_text_semantic(fp, **options):
    # type: (Path|str, Any) -> Dict[str, Any]
    """
    Generate ISCC Semantic-Code Text from a text file.

    NOTE:
        If you enable generating granular features with `features=True` those features will have
        the same bit-length as the generated ISCC-UNIT.

    :param fp: File path of plaintext file to process
    :param options: Custom processing options for overriding global options
    :key bits (int): Length of generated Semantic Text-Code in bits (Default 64)
    :key characters (bool): Return document character count (Default True).
    :key features (bool): Return granular document features (Default False).
    :key offsets (bool): Return character offsets for granular features (Default False).
    :key chunks (bool): Return text chunks (Default False).
    :return: Dict with ISCC processing results
    """
    return gen_text_code_semantic(fp.read_text(encoding="utf-8"), **options)


def gen_text_code_semantic(text, **options):
    # type: (str, int) -> dict
    """
    Create an ISCC Semantic-Code Text from plaintext.

    :param str text: Plaint text for ISCC processing
    :param options: Custom processing options for overriding global options
    :key bits (int): Length of generated Semantic Text-Code in bits (Default 64)
    :key characters (bool): Return document character count (Default True).
    :key features (bool): Return granular document features (Default False).
    :key offsets (bool): Return character offsets for granular features (Default False).
    :key chunks (bool): Return text chunks (Default False).
    :return: Dict with ISCC processing results
    """
    opts = sct.SctOptions(**options) if options else sct.sct_opts

    result = {"iscc": None}  # Initialize first so `iscc` key is first in dict

    if opts.characters:
        result["characters"] = len(text)

    # Text splitting
    splits = split_text(text)
    offsets, chunks = [list(item) for item in zip(*splits)]
    if opts.chunks:
        result["chunks"] = chunks
    if opts.offsets:
        result["offsets"] = offsets

    # Chunk embedding
    embeddings = embed_chunks(chunks)
    if opts.features:
        feature_digests = [binarize(vec)[: opts.bits // 8] for vec in embeddings]
        result["features"] = [sct.encode_base32(digest) for digest in feature_digests]

    # Create global document embedding
    embedding = mean_pooling(embeddings)
    if opts.embedding:
        result["embedding"] = embedding.tolist()

    # Encode global document embedding
    mtype = "0001"  # SEMANTIC
    stype = "0000"  # TEXT
    version = "0000"  # V0
    length = BIT_LEN_MAP[opts.bits]
    header = int(mtype + stype + version + length, 2).to_bytes(2, byteorder="big")
    digest = binarize(embedding)[: opts.bits // 8]
    code = sct.encode_base32(header + digest)
    result["iscc"] = "ISCC:" + code
    return result


def split_text(text):
    # type: (str) -> List[Tuple[int,str]]
    """
    Split text into semantically coherent chunks for embedding.

    :param str text: Text to split.
    :return: A list of offset, chunk tuples [(offset,chunk), ...]
    """
    return splitter().chunk_indices(text)


@cache
def tokenizer():
    # type: () -> Tokenizer
    """
    Load and cache the tokenizer model based on the predefined model name.

    :return: An instance of the Tokenizer.
    :rtype: Tokenizer
    """
    with sct.timer("TOKENIZER load time"):
        return Tokenizer.from_file(TOKENIZER_PATH.as_posix())


@cache
def splitter():
    # type: () -> TextSplitter
    """
    Load and cache the text splitter, initialized with a tokenizer.

    :return: An instance of TextSplitter.
    :rtype: TextSplitter
    """
    with sct.timer("TEXTSPLITTER load time"):
        return TextSplitter.from_huggingface_tokenizer(
            tokenizer(), capacity=CAPACITY, overlap=OVERLAP, trim=TRIM
        )


@cache
def model():
    # type: () -> rt.InferenceSession
    """
    Load and cache the ONNX inference model from a specified path.

    :return: An ONNX inference session.
    :rtype: rt.InferenceSession
    """
    with sct.timer("ONNXMODEL aquisition time"):
        model_path = sct.get_model()
    available_onnx_providers = rt.get_available_providers()
    log.debug(f"Available ONNX providers {', '.join(available_onnx_providers)}")
    selected_onnx_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_onnx_providers:
        selected_onnx_providers.insert(0, "CUDAExecutionProvider")
    log.debug(f"Using ONNX providers {', '.join(selected_onnx_providers)}")
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    with sct.timer("ONNXMODEL load time"):
        return rt.InferenceSession(model_path, sess_options=so, providers=selected_onnx_providers)


def tokenize_chunks(chunks):
    # type: (List[str]) -> dict
    """
    Tokenize text chunks into model-compatible formats.

    :param List[str] chunks: Text chunks to tokenize.
    :return: Dictionary of tokenized data including input IDs, attention masks, and type IDs.
    :rtype: dict
    """
    encodings = tokenizer().encode_batch(chunks)
    input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
    type_ids = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": type_ids}


def embed_chunks(chunks):
    # type: (List[str]) -> NDArray[np.float32]
    """
    Embed text chunks and return vector embeddings.

    :param List[str] chunks: Text chunks to embed.
    :return: An array of embeddings for each chunk.
    :rtype: NDArray[np.float32]
    """
    tokens = tokenize_chunks(chunks)
    token_embeddings = embed_tokens(tokens)
    attention_pooling(token_embeddings, tokens["attention_mask"])
    return attention_pooling(token_embeddings, tokens["attention_mask"])


def embed_tokens(tokens):
    # type: (dict) -> NDArray
    """
    Create embeddings from tokenized text chunks using the model.

    :param dict tokens: Tokenized text data.
    :return: An array of embeddings.
    :rtype: NDArray
    """
    result = model().run(None, tokens)
    return np.array(result[0])


def attention_pooling(token_embeddings, attention_mask):
    # type: (np.array, np.array) -> np.array
    """
    Apply attention mask based mean pooling to the token embeddings.

    :param np.array token_embeddings: Raw token embeddings from the model.
    :param np.array attention_mask: Attention masks for the embeddings.
    :return: An array of pooled and normalized embeddings.
    :rtype: np.array
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

    :param NDArray[np.float32] embeddings: Chunk embeddings.
    :return: A normalized document vector.
    :rtype: NDArray
    """
    document_vector = embeddings.mean(axis=0)
    return document_vector / np.linalg.norm(document_vector)


def binarize(vec):
    # type: (NDArray) -> bytes
    """
    Binarize an embedding vector into a hash digest.

    :param NDArray vec: Vector to be binarized.
    :return: A bytes object representing the binary hash.
    :rtype: bytes
    """
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
