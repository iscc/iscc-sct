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

from importlib.metadata import PackageNotFoundError, distribution
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from functools import cache, partial
import re
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

# Newline runs that can act as paragraph-level split boundaries (2+ newline characters)
NEWLINE_RUNS = re.compile(r"[\r\n]{2,}")

# A single newline character - used by needs_split_guard to find separator-free spans
NEWLINE = re.compile(r"[\r\n]")

# Any Unicode whitespace - used by token_count_guarded to find a tokenizer word boundary.
# The tokenizer's WhitespaceSplit pre-tokenizer splits on Unicode whitespace (not just ASCII),
# so the prefix cut must match that to short-circuit NBSP/em-space/form-feed PDF text.
WHITESPACE = re.compile(r"\s")

# Max distance (chars) from any position to the next paragraph-level separator before
# chunking switches to the guarded splitter (see needs_split_guard)
SPLIT_GUARD_GAP = 8192


def code_text_semantic(fp, **options):
    # type: (Path|str, Any) -> dict[str, Any]
    """
    Generate ISCC Semantic-Code Text from a text file.

    NOTE:
        Enable granular features with `simprints=True`. Their length is set by `bits_granular`
        (default 64) and is independent of the document `bits`.

    :param fp: File path of a plaintext file to process
    :param options: Custom processing options for overriding global options. Recognized keys:

        - ``bits`` (int): Length of generated Semantic Text-Code in bits (default 64).
        - ``characters`` (bool): Return document character count (default True).
        - ``embedding`` (bool): Return global document embedding (default False).
        - ``precision`` (int): Max fractional digits for embeddings (default 8).
        - ``simprints`` (bool): Return granular document features (default False).
        - ``offsets`` (bool): Return character offsets for granular features (default False).
        - ``sizes`` (bool): Include sizes of granular features in chars (default False).
        - ``contents`` (bool): Return text chunks (default False).
        - ``max_tokens`` (int): Max tokens per chunk (default 127).
        - ``overlap`` (int): Max tokens allowed to overlap between chunks (default 48).
        - ``trim`` (bool): Trim whitespace from chunks (default False).
    :return: Dict with ISCC processing results
    """
    fp = Path(fp)
    return gen_text_code_semantic(fp.read_text(encoding="utf-8"), **options)


def gen_text_code_semantic(text, **options):
    # type: (str, Any) -> dict
    """
    Create an ISCC Semantic-Code Text from plaintext.

    :param str text: Plain text for ISCC processing
    :param options: Custom processing options for overriding global options. Recognized keys:

        - ``bits`` (int): Length of generated Semantic Text-Code in bits (default 64).
        - ``characters`` (bool): Return document character count (default True).
        - ``embedding`` (bool): Return global document embedding (default False).
        - ``precision`` (int): Max fractional digits for embeddings (default 8).
        - ``simprints`` (bool): Return granular document features (default False).
        - ``offsets`` (bool): Return character offsets for granular features (default False).
        - ``sizes`` (bool): Include sizes of granular features in chars (default False).
        - ``contents`` (bool): Return text chunks (default False).
        - ``max_tokens`` (int): Max tokens per chunk (default 127).
        - ``overlap`` (int): Max tokens allowed overlapping between chunks (default 48).
        - ``trim`` (bool): Trim whitespace from chunks (default False).
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
        embeddings = embed_chunks(chunks)

    # Create global document embedding
    embedding = mean_pooling(embeddings)

    if any([opts.simprints, opts.offsets, opts.sizes, opts.contents, opts.embedding]):
        feature_set = {
            "maintype": "semantic",
            "subtype": "text",
            "version": 0,
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

    # Encode global document embedding
    length = BIT_LEN_MAP[opts.bits]
    header = int(MAINTYPE + SUBTYPE + SCT_VERSION + length, 2).to_bytes(2, byteorder="big")
    digest = binarize(embedding)[: opts.bits // 8]
    code = sct.encode_base32(header + digest)
    result["iscc"] = "ISCC:" + code
    return result


def soft_hash_text_semantic(text):
    # type: (str) -> bytes
    """Create a similarity-preserving hash for text as the full binarized document embedding (384 bits)."""
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
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: A list of offset, chunk tuples [(offset, chunk), ...]
    """
    opts = sct.sct_opts.override(options)
    select = splitter_guarded if needs_split_guard(text) else splitter
    chunks = select(**opts.model_dump()).chunk_indices(text)

    if not opts.byte_offsets:
        return chunks

    # Convert character offsets to byte offsets
    char_positions = [offset for offset, _ in chunks]
    byte_positions = sct.char_to_byte_offsets(text, char_positions)

    return [(byte_positions[i], chunk) for i, (_, chunk) in enumerate(chunks)]


@cache
def tokenizer():
    # type: () -> Tokenizer
    """
    Load and cache the tokenizer model based on the predefined model name.

    This tokenizer keeps the vendored truncation (128 tokens) and padding settings and is used
    to embed chunks. For chunk sizing use chunking_tokenizer() instead.

    :return: An instance of the Tokenizer.
    """
    with sct.timer("TOKENIZER load time"):
        return Tokenizer.from_file(TOKENIZER_PATH.as_posix())


@cache
def chunking_tokenizer():
    # type: () -> Tokenizer
    """
    Load and cache the tokenizer used for chunk sizing, with truncation and padding disabled.

    The embedding tokenizer truncates to the model's 128-token window. With tokenizers >=0.23
    the Hugging Face chunk sizer then sees one overflow encoding per 128 tokens, so sizing a
    huge probe string costs O(length) and chunking degrades to super-linear runtime (issue
    #24). Chunk sizing needs the true token count, so truncation is disabled here; padding
    would only add tokens irrelevant to a count and is disabled too. Boundaries stay unchanged
    because the splitter only compares sizes against the token capacity: at or below it the
    full and truncated counts agree, and any larger probe exceeds it under both, so every
    accept/reject decision is identical.

    :return: A Tokenizer with truncation and padding disabled.
    """
    tok = Tokenizer.from_file(TOKENIZER_PATH.as_posix())
    tok.no_truncation()
    tok.no_padding()
    return tok


@cache
def splitter(**options):
    # type: (Any) -> TextSplitter
    """
    Load and cache the text splitter, initialized with the chunking tokenizer.

    :param options: Custom processing options for overriding global options
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: An instance of TextSplitter.
    """
    opts = sct.sct_opts.override(options)
    with sct.timer("TEXTSPLITTER load time"):
        return TextSplitter.from_huggingface_tokenizer(
            chunking_tokenizer(), capacity=opts.max_tokens, overlap=opts.overlap, trim=opts.trim
        )


@cache
def splitter_guarded(**options):
    # type: (Any) -> TextSplitter
    """
    Load and cache a text splitter that sizes chunks via a guarded Python callback.

    Produces chunks identical to splitter() but avoids the super-linear cost of sizing huge
    probe texts on inputs without regular paragraph separators (see needs_split_guard).

    :param options: Custom processing options for overriding global options
    :key max_tokens (int): Max tokens per chunk (default 127).
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :key trim (int): Trim whitespace from chunks (default False).
    :return: An instance of TextSplitter.
    """
    opts = sct.sct_opts.override(options)
    sizer = partial(token_count_guarded, max_tokens=opts.max_tokens)
    with sct.timer("TEXTSPLITTER load time"):
        return TextSplitter.from_callback(
            sizer, capacity=opts.max_tokens, overlap=opts.overlap, trim=opts.trim
        )


def needs_split_guard(text):
    # type: (str) -> bool
    """
    Detect text where tokenizer-based chunking degrades to super-linear runtime.

    To size an oversized section, text-splitter (>=0.32.0) probes prefixes up to that
    section's lower-level semantic boundaries instead of tokenizing the whole section, which
    keeps chunking near-linear while such boundaries exist. When a span carries no
    intermediate separator for more than SPLIT_GUARD_GAP characters - a giant single
    paragraph, a trailing separator-free run, or words sitting far from the next
    paragraph-level separator (print-layout PDF extraction, issue #24) - the probe finds no
    boundary and falls back to tokenizing the whole section, so chunking time grows
    quadratically with the gap size.

    Such texts are routed to the guarded splitter, whose token sizer caps that fallback cost.

    :param text: Text to analyze.
    :return: True if the guarded splitter should be used for this text.
    """
    runs = []
    for match in NEWLINE_RUNS.finditer(text):
        run = match.group()
        level = len(run) - run.count("\r\n")  # number of newline graphemes in the run
        if level >= 2:
            runs.append((level, match.start(), match.end()))
    for min_level in sorted({level for level, _, _ in runs}):
        pos = 0
        for level, start, end in runs:
            if level < min_level:
                continue
            if start - pos > SPLIT_GUARD_GAP:
                return True
            pos = end
    # A span containing no newline at all blows up the native sizer the same way: with no
    # line- or paragraph-level separator to bound the probes, sizing reaches toward a distant
    # or absent coarse separator. Catches giant single paragraphs and trailing separator-free
    # runs that the paragraph-level scan above does not see.
    pos = 0
    for match in NEWLINE.finditer(text):
        if match.start() - pos > SPLIT_GUARD_GAP:
            return True
        pos = match.end()
    return len(text) - pos > SPLIT_GUARD_GAP


def token_count(text):
    # type: (str) -> int
    """
    Count tokens exactly like text-splitter's Hugging Face tokenizer chunk sizer.

    Encodes without special tokens via the chunking tokenizer (truncation and padding
    disabled), so the count reflects the full input text with no overflow encodings to sum.

    :param text: Text to size.
    :return: Number of tokens.
    """
    return len(chunking_tokenizer().encode(text, add_special_tokens=False).ids)


def token_count_guarded(text, max_tokens):
    # type: (str, int) -> int
    """
    Count tokens with a short-circuit for texts far larger than the chunk capacity.

    For long texts, tokenize only a prefix that ends at a Unicode whitespace boundary. The
    tokenizer pre-splits on whitespace (WhitespaceSplit + Metaspace), so the full text has at
    least as many tokens as that prefix. If the prefix alone exceeds max_tokens, full
    tokenization is skipped - the splitter only needs to know that the text is too big for one
    chunk. The returned overestimate stays above max_tokens and grows with text length, so all
    chunk capacity comparisons behave exactly as with real token counts.

    :param text: Text to size.
    :param max_tokens: Chunk capacity the splitter validates against.
    :return: Number of tokens (exact, or an overestimate for oversized texts).
    """
    probe_chars = max_tokens * 10
    if len(text) > probe_chars * 2:
        window = text[:probe_chars]
        cut = -1
        for match in WHITESPACE.finditer(window):
            cut = match.start()
        if cut > 0:
            prefix_count = token_count(text[:cut])
            if prefix_count > max_tokens:
                return prefix_count + len(text) - cut
    return token_count(text)


def warn_gpu_shadowed(available_providers):
    # type: (list[str]) -> None
    """
    Warn when onnxruntime-gpu is installed but CUDA support is unavailable.

    Both onnxruntime variant wheels unpack into the same directory, so installing the CPU
    package alongside onnxruntime-gpu silently disables CUDA support.

    :param available_providers: Providers reported by the installed onnxruntime build.
    """
    if "CUDAExecutionProvider" in available_providers:
        return
    try:
        distribution("onnxruntime-gpu")
    except PackageNotFoundError:
        return
    log.warning(
        "onnxruntime-gpu is installed but CUDA support is unavailable - the onnxruntime CPU "
        "package likely overwrote the GPU build. To fix run: pip uninstall -y onnxruntime "
        'onnxruntime-gpu && pip install --force-reinstall "iscc-sct[gpu]"'
    )


ONNX_RUNTIME_MISSING = (
    "iscc-sct requires an ONNX runtime. Install exactly one of:\n"
    '  pip install "iscc-sct[cpu]"  # CPU inference\n'
    '  pip install "iscc-sct[gpu]"  # NVIDIA CUDA accelerated inference'
)


def load_onnxruntime():
    # type: () -> Any
    """
    Import and return the onnxruntime module.

    The ONNX runtime is an optional dependency selected via the mutually exclusive cpu/gpu
    extras, so the import is deferred until a model is actually needed. This lets the rest of
    the package - and the `iscc-sct doctor` command - load without a runtime installed.

    :return: The imported onnxruntime module.
    :raises ImportError: If no ONNX runtime is installed.
    """
    try:
        import onnxruntime as rt
    except ImportError:  # pragma: no cover - exercised only without a runtime extra
        raise ImportError(ONNX_RUNTIME_MISSING) from None
    return rt


@cache
def model():
    # type: () -> Any
    """
    Load and cache the ONNX inference model from a specified path.

    :return: An ONNX inference session (onnxruntime.InferenceSession).
    """
    rt = load_onnxruntime()
    from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

    available_onnx_providers = rt.get_available_providers()
    log.debug(f"Available ONNX providers {', '.join(available_onnx_providers)}")
    warn_gpu_shadowed(available_onnx_providers)
    selected_onnx_providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_onnx_providers:  # pragma: no cover
        selected_onnx_providers.insert(0, "CUDAExecutionProvider")
        if hasattr(rt, "preload_dlls"):
            # Load CUDA/cuDNN libraries from pip-provided nvidia wheels or the system PATH
            # before session creation (available since onnxruntime 1.21).
            rt.preload_dlls()
    log.debug(f"Using ONNX providers {', '.join(selected_onnx_providers)}")
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        with sct.timer("ONNXMODEL load time"):
            return rt.InferenceSession(
                sct.MODEL_PATH, sess_options=so, providers=selected_onnx_providers
            )
    except NoSuchFile:  # pragma: no cover
        with sct.timer("ONNXMODEL aquisition/load time"):
            model_path = sct.get_model()
            return rt.InferenceSession(
                model_path, sess_options=so, providers=selected_onnx_providers
            )


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


def embed_chunks(chunks, batch_size=100):
    """
    Embed text chunks and return vector embeddings.

    :param chunks: Text chunks to embed.
    :param batch_size: Number of chunks to process in each batch.
    :return: An array of embeddings for each chunk.
    """
    embeddings = []
    for start_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start_idx : start_idx + batch_size]
        tokens = tokenize_chunks(batch_chunks)
        token_embeddings = embed_tokens(tokens)
        batch_embeddings = attention_pooling(token_embeddings, tokens["attention_mask"])
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


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
