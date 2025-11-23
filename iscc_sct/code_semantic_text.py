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
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from pathlib import Path
from enum import Enum
import numpy as np
import onnxruntime as rt
from functools import cache
import iscc_sct as sct
from iscc_sct.models_config import get_model_config


HERE = Path(__file__).parent.absolute()


class EmbeddingGemmaPrompt(Enum):
    """
    Prompt types for EmbeddingGemma model as specified in the official documentation.
    See: https://huggingface.co/blog/embeddinggemma

    These prompts are task-specific and were used during model training.
    """

    # Primary prompts for common use cases
    DOCUMENT = "title: none | text: "  # For document embedding (Retrieval-document)
    QUERY = "task: search result | query: "  # For search queries (Retrieval-query, Retrieval, Reranking)

    # Task-specific prompts
    BITEXT_MINING = "task: search result | query: "  # For bilingual text mining
    CLUSTERING = "task: clustering | query: "  # For clustering tasks
    CLASSIFICATION = "task: classification | query: "  # For classification tasks
    INSTRUCTION_RETRIEVAL = "task: code retrieval | query: "  # For code/instruction retrieval
    MULTILABEL_CLASSIFICATION = "task: classification | query: "  # For multi-label classification
    PAIR_CLASSIFICATION = "task: sentence similarity | query: "  # For sentence pair classification
    RERANKING = "task: search result | query: "  # For re-ranking search results
    RETRIEVAL = "task: search result | query: "  # General retrieval tasks
    RETRIEVAL_QUERY = "task: search result | query: "  # Explicit query retrieval
    RETRIEVAL_DOCUMENT = "title: none | text: "  # Explicit document retrieval
    STS = "task: sentence similarity | query: "  # Semantic Textual Similarity
    SUMMARIZATION = "task: summarization | query: "  # For summarization tasks

    # Special case
    NONE = ""  # No prompt (for backward compatibility or other models)


__all__ = [
    "code_text_semantic",
    "gen_text_code_semantic",
    "soft_hash_text_semantic",
    "embed_chunks",
    "EmbeddingGemmaPrompt",
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

    :param fp: File path of a plaintext file to process
    :param options: Custom processing options for overriding global options
    :key bits (int): Length of generated Semantic Text-Code in bits (default 64)
    :key characters (bool): Return document character count (default True).
    :key embedding (bool): Return global document embedding (default False).
    :key precision (int): Max fractional digits for embeddings (default 8).
    :key simprints (bool): Return granular document features (default False).
    :key offsets (bool): Return character offsets for granular features (default False).
    :key sizes (bool): Include sizes of granular features (number of chars, default False).
    :key contents (bool): Return text chunks (default False).
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
    :key model_version (int): Model version to use (default from sct_opts).
    :key prompt_type (str|EmbeddingGemmaPrompt): Prompt type for EmbeddingGemma (default 'DOCUMENT').
    :key bits (int): Length of generated Semantic Text-Code in bits (default 64)
    :key characters (bool): Return document character count (default True).
    :key embedding (bool): Return global document embedding (default False).
    :key precision (int): Max fractional digits for embeddings (default 8).
    :key simprints (bool): Return granular document features (default False).
    :key offsets (bool): Return character offsets for granular features (default False).
    :key sizes (bool): Include sizes of granular features (number of chars, default False).
    :key contents (bool): Return text chunks (default False).
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: Dict with ISCC processing results (using Index-Format for granular features)
    """

    if not text:
        raise ValueError("Input text cannot be empty.")

    opts = sct.sct_opts.override(options)

    result = {"iscc": None}  # Initialize first so `iscc` key is "first" in dict

    if opts.characters:
        result["characters"] = len(text)

    # Text splitting
    splits = split_text(text, **opts.model_dump())
    offsets, chunks = [list(item) for item in zip(*splits)]

    # Chunk embedding
    with sct.timer("EMBEDDING time"):
        embeddings = embed_chunks(
            chunks, model_version=opts.model_version, prompt_type=opts.prompt_type
        )

    # Create global document embedding
    embedding = mean_pooling(embeddings)

    if any([opts.simprints, opts.offsets, opts.sizes, opts.contents, opts.embedding]):
        feature_set = {
            "maintype": "semantic",
            "subtype": "text",
            "version": opts.model_version,
        }
        if opts.offsets or opts.sizes:
            feature_set["byte_offsets"] = opts.byte_offsets
        if opts.embedding:
            feature_set["embedding"] = compress(embedding, opts.precision)
        if opts.simprints:
            feature_digests = [binarize(vec)[: opts.bits_granular // 8] for vec in embeddings]
            feature_set["simprints"] = [sct.encode_base64(digest) for digest in feature_digests]
        if opts.offsets:
            feature_set["offsets"] = offsets
        if opts.sizes:
            if opts.byte_offsets:
                feature_set["sizes"] = [len(chunk.encode("utf-8")) for chunk in chunks]
            else:
                feature_set["sizes"] = [len(chunk) for chunk in chunks]
        if opts.contents:
            feature_set["contents"] = chunks
        result["features"] = [feature_set]

    # Encode global document embedding with model version in header
    length = BIT_LEN_MAP[opts.bits]
    version_bits = format(opts.model_version, "04b")  # Convert version to 4-bit binary string
    header = int(MAINTYPE + SUBTYPE + version_bits + length, 2).to_bytes(2, byteorder="big")
    digest = binarize(embedding)[: opts.bits // 8]
    code = sct.encode_base32(header + digest)
    result["iscc"] = "ISCC:" + code
    return result


def soft_hash_text_semantic(text, model_version=0, prompt_type=None):
    # type: (str, int, EmbeddingGemmaPrompt|str|None) -> bytes
    """
    Creates a 256-bit semantic similarity-preserving hash for text input.

    :param text: Text to hash.
    :param model_version: Model version to use (default 0).
    :param prompt_type: Prompt type for EmbeddingGemma (default None, auto-selects based on model).
    :return: 256-bit binary hash digest.
    """
    chunks = [item[1] for item in split_text(text, model_version=model_version)]
    embeddings = embed_chunks(chunks, model_version=model_version, prompt_type=prompt_type)
    embedding = mean_pooling(embeddings)
    digest = binarize(embedding)
    return digest


def split_text(text, **options):
    # type: (str, Any) -> list[tuple[int,str]]
    """
    Split text into semantically coherent chunks for embedding.

    :param text: Text to split.
    :param options: Custom processing options for overriding global options
    :key model_version (int): Model version to use (default from sct_opts).
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: A list of offset, chunk tuples [(offset, chunk), ...]
    """
    opts = sct.sct_opts.override(options)
    chunks = splitter(
        model_version=opts.model_version,
        max_tokens=opts.max_tokens,
        overlap=opts.overlap,
        trim=opts.trim,
    ).chunk_indices(text)

    if not opts.byte_offsets:
        return chunks

    # Convert character offsets to byte offsets
    char_positions = [offset for offset, _ in chunks]
    byte_positions = sct.char_to_byte_offsets(text, char_positions)

    return [(byte_positions[i], chunk) for i, (_, chunk) in enumerate(chunks)]


@cache
def tokenizer(model_version=0):
    # type: (int) -> Tokenizer
    """
    Load and cache the tokenizer for the specified model version.

    :param model_version: Model version integer (default 0)
    :return: An instance of the Tokenizer.
    """
    with sct.timer("TOKENIZER load time"):
        if model_version == 0:
            # Version 0 uses the bundled tokenizer in the package
            tokenizer_path = TOKENIZER_PATH
        else:
            # Other versions download tokenizer.json alongside model files
            model_dir = sct.get_model_path(model_version)
            tokenizer_path = model_dir / "tokenizer.json"
            # Ensure model files (including tokenizer) are downloaded
            if not tokenizer_path.exists():
                sct.get_model(model_version)

        tok = Tokenizer.from_file(tokenizer_path.as_posix())

        # Enable padding for model versions that need it (v1+)
        # This ensures all sequences have the same length when creating numpy arrays
        # Model v0 has padding pre-configured in its tokenizer.json
        if model_version >= 1:
            tok.enable_padding()

        return tok


@cache
def splitter(model_version=0, max_tokens=127, overlap=48, trim=False):
    # type: (int, int, int, bool) -> TextSplitter
    """
    Load and cache the text splitter, initialized with tokenizer.

    :param model_version: Model version integer (default 0)
    :param max_tokens: Max tokens per chunk (default 127).
    :param overlap: Max tokens allowed overlapping between chunks (default 48).
    :param trim: Trim whitespace from chunks (default False).
    :return: An instance of TextSplitter.
    """
    with sct.timer("TEXTSPLITTER load time"):
        return TextSplitter.from_huggingface_tokenizer(
            tokenizer(model_version), capacity=max_tokens, overlap=overlap, trim=trim
        )


@cache
def model(model_version=0):
    # type: (int) -> rt.InferenceSession
    """
    Load and cache the ONNX inference model for the specified version.

    :param model_version: Model version integer (default 0)
    :return: An ONNX inference session.
    """
    config = get_model_config(model_version)
    model_dir = sct.get_model_path(model_version)

    available_onnx_providers = rt.get_available_providers()
    log.debug(f"Available ONNX providers {', '.join(available_onnx_providers)}")
    selected_onnx_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_onnx_providers:  # pragma: no cover
        selected_onnx_providers.insert(0, "CUDAExecutionProvider")
    log.debug(f"Using ONNX providers {', '.join(selected_onnx_providers)}")
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Primary model file is always the first file in the list
    model_file = model_dir / config.filenames[0]

    try:
        with sct.timer("ONNXMODEL load time"):
            return rt.InferenceSession(
                model_file, sess_options=so, providers=selected_onnx_providers
            )
    except NoSuchFile:  # pragma: no cover
        with sct.timer("ONNXMODEL acquisition/load time"):
            sct.get_model(model_version)
            return rt.InferenceSession(
                model_file, sess_options=so, providers=selected_onnx_providers
            )


def tokenize_chunks(chunks, model_version=0, prompt_type=None):
    # type: (list[str], int, EmbeddingGemmaPrompt|str|None) -> dict
    """
    Tokenize text chunks into model-compatible formats.

    :param chunks: Text chunks to tokenize.
    :param model_version: Model version to use (default 0).
    :param prompt_type: Prompt type for EmbeddingGemma (default None, auto-selects based on model).
    :return: Dictionary of tokenized data with model-specific inputs.
    """
    # Apply task-specific prompts for EmbeddingGemma (model version 1)
    if model_version == 1:
        # Handle different prompt_type input formats
        if isinstance(prompt_type, EmbeddingGemmaPrompt):
            # Already an enum, use as is
            pass
        elif isinstance(prompt_type, str):
            # Convert string to enum
            try:
                prompt_type = EmbeddingGemmaPrompt[prompt_type.upper()]
            except KeyError:
                log.warning(f"Invalid prompt_type '{prompt_type}', using default DOCUMENT prompt")
                prompt_type = EmbeddingGemmaPrompt.DOCUMENT
        elif prompt_type is None:
            # Use DOCUMENT as default for model version 1
            prompt_type = EmbeddingGemmaPrompt.DOCUMENT
        else:
            log.warning(
                f"Unexpected prompt_type type '{type(prompt_type)}', using default DOCUMENT prompt"
            )
            prompt_type = EmbeddingGemmaPrompt.DOCUMENT

        # Apply the prompt prefix to each chunk
        if prompt_type != EmbeddingGemmaPrompt.NONE:
            chunks = [f"{prompt_type.value}{chunk}" for chunk in chunks]
            log.debug(f"Applied EmbeddingGemma prompt: {prompt_type.name}")

    tok = tokenizer(model_version)
    encodings = tok.encode_batch(chunks)
    input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)

    # Version 0 needs token_type_ids, version 1 (embeddinggemma) does not
    if model_version == 0:
        type_ids = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }
    else:
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def embed_chunks(chunks, batch_size=100, model_version=0, prompt_type=None):
    # type: (list[str], int, int, EmbeddingGemmaPrompt|str|None) -> NDArray
    """
    Embed text chunks and return vector embeddings.

    :param chunks: Text chunks to embed.
    :param batch_size: Number of chunks to process in each batch.
    :param model_version: Model version to use (default 0).
    :param prompt_type: Prompt type for EmbeddingGemma (default None, auto-selects based on model).
    :return: An array of embeddings for each chunk.
    """
    embeddings = []
    for start_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start_idx : start_idx + batch_size]
        tokens = tokenize_chunks(batch_chunks, model_version, prompt_type)
        token_embeddings = embed_tokens(tokens, model_version)
        batch_embeddings = attention_pooling(token_embeddings, tokens["attention_mask"])
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def embed_tokens(tokens, model_version=0):
    # type: (dict, int) -> NDArray
    """
    Create embeddings from tokenized text chunks using the model.

    :param tokens: Tokenized text data.
    :param model_version: Model version to use (default 0).
    :return: An array of embeddings.
    """
    result = model(model_version).run(None, tokens)
    return np.array(result[0])


def attention_pooling(token_embeddings, attention_mask):
    # type: (np.array, np.array) -> np.array
    """
    Apply attention mask-based mean pooling to the token embeddings.

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
    :return: A byte object representing the binary hash.
    """
    return bytes((np.packbits(np.array(vec) >= 0)))


def compress(vec, precision):
    # type: (NDArray, int) -> list[float]
    """
    Round down vector values to specified precision to reduce storage requirements.

    :param vec: Embedding vector.
    :param precision: Max number of fractional decimal places.
    :return: Vector as a native python list of rounded floats.
    """
    rounded_array = np.around(vec, decimals=precision)
    compress_list = [round(x, precision) for x in rounded_array.tolist()]
    return compress_list
