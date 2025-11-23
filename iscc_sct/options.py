from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from iscc_sct.models_config import MODEL_REGISTRY


__all__ = [
    "SctOptions",
    "sct_opts",
]


load_dotenv()


class SctOptions(BaseSettings):
    bits: int = Field(
        64,
        description="ISCC_SCT_BITS - Default bit-length of generated Semantic Text-Code in bits",
        ge=32,
        le=256,
        multiple_of=32,
    )

    bits_granular: int = Field(
        64,
        description="ISCC_SCT_BITS_GRANULAR - Default bit-length of granular features",
        ge=32,
        le=256,
        multiple_of=32,
    )

    characters: bool = Field(
        True, description="ISCC_SCT_CHARACTERS - Include document character count"
    )
    embedding: bool = Field(
        False, description="ISCC_SCT_EMBEDDING - Include global document embedding"
    )

    precision: int = Field(
        8, description="ISCC_SCT_PRECISION - Max fractional digits for embeddings (default 8)"
    )

    simprints: bool = Field(
        False, description="ISCC_SCT_SIMPRINTS - Include granular feature simprints"
    )
    offsets: bool = Field(
        False, description="ISCC_SCT_OFFSETS - Include offsets of granular features"
    )

    byte_offsets: bool = Field(
        False,
        description="ISCC_SCT_BYTE_OFFSETS - Use UTF-8 byte offsets instead of character offsets",
    )

    sizes: bool = Field(
        False, description="ISCC_SCT_SIZES - Include sizes of granular features (number of chars)"
    )

    contents: bool = Field(False, description="ISCC_SCT_CONTENTS - Include granular text chunks")

    max_tokens: int = Field(
        127,
        description="ISCC_SCT_MAX_TOKENS - Max tokens per chunk (Default 127, max 2048)",
        le=2048,
    )

    overlap: int = Field(
        48,
        description="ISCC_SCT_OVERLAP - Max tokens allowed to overlap between chunks (Default 48)",
    )

    trim: bool = Field(
        False, description="ISCC_SCT_TRIM - Trim whitespace from chunks (Default False)"
    )

    download_timeout: int = Field(
        600,
        description="ISCC_SCT_DOWNLOAD_TIMEOUT - Timeout in seconds for model download lock acquisition (Default 600)",
        gt=0,
    )

    model_dir: str | None = Field(
        None,
        description="ISCC_SCT_MODEL_DIR - Custom directory for model storage (default: platform-specific user data dir)",
    )

    model_version: int = Field(
        0,
        description="ISCC_SCT_MODEL_VERSION - Embedding model version (0=minilm-l12, 1=embeddinggemma-300m)",
        ge=0,
    )

    prompt_type: str | None = Field(
        None,
        description="ISCC_SCT_PROMPT_TYPE - EmbeddingGemma prompt type (e.g., DOCUMENT, QUERY, CLUSTERING)",
    )

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, v):
        # type: (int) -> int
        """Validate that model_version exists in MODEL_REGISTRY."""
        if v not in MODEL_REGISTRY:
            available = ", ".join(str(ver) for ver in sorted(MODEL_REGISTRY.keys()))
            raise ValueError(f"Invalid model_version {v}. Available versions: {available}")
        return v

    @field_validator("prompt_type", mode="before")
    @classmethod
    def validate_prompt_type(cls, v):
        # type: (str|None) -> str|None
        """Validate prompt_type is a valid EmbeddingGemmaPrompt value."""
        if v is None:
            return v

        # Handle EmbeddingGemmaPrompt enum objects
        try:
            from iscc_sct.code_semantic_text import EmbeddingGemmaPrompt

            if isinstance(v, EmbeddingGemmaPrompt):
                return v.name
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

        # Handle string values - convert to uppercase for case-insensitive comparison
        if isinstance(v, str):
            v_upper = v.upper()
            valid_prompts = [
                "DOCUMENT",
                "QUERY",
                "BITEXT_MINING",
                "CLUSTERING",
                "CLASSIFICATION",
                "INSTRUCTION_RETRIEVAL",
                "MULTILABEL_CLASSIFICATION",
                "PAIR_CLASSIFICATION",
                "RERANKING",
                "RETRIEVAL",
                "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT",
                "STS",
                "SUMMARIZATION",
                "NONE",
            ]
            if v_upper not in valid_prompts:
                raise ValueError(
                    f"Invalid prompt_type '{v}'. Valid options: {', '.join(valid_prompts)}"
                )
            return v_upper

        # Try to handle any object with a 'name' attribute (like enum)
        if hasattr(v, "name"):
            return v.name

        raise ValueError(
            f"prompt_type must be a string or EmbeddingGemmaPrompt enum, got {type(v)}"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ISCC_SCT_",
        extra="ignore",
        validate_assignment=True,
    )

    def override(self, update=None):
        # type: (dict|None) -> SctOptions
        """Returns an updated and validated deep copy of the current settings instance."""

        update = update or {}  # sets {} if update is None

        opts = self.model_copy(deep=True)
        # We need update fields individually so validation gets triggered
        for field, value in update.items():
            setattr(opts, field, value)
        return opts


sct_opts = SctOptions()
