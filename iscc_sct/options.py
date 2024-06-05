from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = [
    "SctOptions",
    "sct_opts",
]


load_dotenv()


class SctOptions(BaseSettings):
    # TODO: This affects the ISCC-UNIT and the granular IDs - Should we have separate options?
    bits: int = Field(
        64,
        description="ISCC_SCT_BITS - Default length of generated Semantic Text-Code in bits",
        ge=32,
        le=256,
        multiple_of=32,
    )

    characters: bool = Field(True, description="ISCC_SCT_CHARACTERS - Include document character count")
    embedding: bool = Field(False, description="ISCC_SCT_EMBEDDING - Include global document embedding")

    precision: int = Field(8, description="ISCC_SCT_PRECISION - Max fractional digits for embeddings (default 8)")

    features: bool = Field(False, description="ISCC_SCT_FEATURES - Include granular feature simprints")
    offsets: bool = Field(False, description="ISCC_SCT_OFFSETS - Include offsets of granular features")

    sizes: bool = Field(False, description="ISCC_SCT_SIZES - Include sizes of granular features (number of chars)")

    chunks: bool = Field(False, description="ISCC_SCT_CHUNKS - Include granular text chunks")

    max_tokens: int = Field(
        127,
        description="ISCC_SCT_MAX_TOKENS - Max tokens per chunk (Default 127)",
        le=127,
    )

    overlap: int = Field(
        48,
        description="ISCC_SCT_OVERLAP - Max tokens allowed to overlap between chunks (Default 48)",
    )

    trim: bool = Field(False, description="ISCC_SCT_TRIM - Trim whitespace from chunks (Default False)")

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
