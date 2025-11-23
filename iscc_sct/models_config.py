"""Model configuration registry for ISCC-SCT embedding models."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    version: int
    name: str
    filenames: list[str]
    urls: list[str]
    checksums: list[str]
    embedding_dim: int
    max_tokens: int
    tokenizer_type: str  # "bundled" or "auto"
    tokenizer_name: str | None = None  # HuggingFace tokenizer name if tokenizer_type="auto"


# Model Registry: Maps version integers to model configurations
MODEL_REGISTRY = {
    0: ModelConfig(
        version=0,
        name="paraphrase-multilingual-minilm-l12-v2",
        filenames=["iscc-sct-v0.1.0.onnx"],
        urls=[
            "https://github.com/iscc/iscc-binaries/releases/download/v1.0.0/iscc-sct-v0.1.0.onnx"
        ],
        checksums=["ff254d62db55ed88a1451b323a66416f60838dd2f0338dba21bc3b8822459abc"],
        embedding_dim=384,
        max_tokens=127,
        tokenizer_type="bundled",
        tokenizer_name=None,
    ),
    1: ModelConfig(
        version=1,
        name="embeddinggemma-300m",
        filenames=["model.onnx", "model.onnx_data", "tokenizer.json"],
        urls=[
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx_data",
            "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.json",
        ],
        checksums=[
            "334f53c543f19693f6a387c4ad09a26fe1e39a1191cf8035430095fecff28277",
            "80d63429b04c4f2c081e2f4bb6b1269b748e38fbe1ebeabd97741977874927dc",
            "6b745bf905a7d920f9ebc1dd7649e76ee8023e2cfe59e4c23a44209a6e91b415",
        ],
        embedding_dim=768,
        max_tokens=2048,
        tokenizer_type="bundled",
        tokenizer_name=None,
    ),
}


def get_model_config(version):
    # type: (int) -> ModelConfig
    """
    Get model configuration for a given version.

    :param version: Model version integer
    :return: ModelConfig for the specified version
    :raises ValueError: If version is not found in registry
    """
    if version not in MODEL_REGISTRY:
        available = ", ".join(str(v) for v in sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Model version {version} not found. Available versions: {available}")
    return MODEL_REGISTRY[version]
