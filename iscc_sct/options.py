from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = [
    "SctOptions",
    "sct_opts",
]


load_dotenv()


class SctOptions(BaseSettings):
    bits: int = Field(
        64,
        description="ISCC_SCT_BITS - Default length of generated Semantic Text-Code in bits",
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
    features: bool = Field(
        False, description="ISCC_SCT_FEATURES - Include granular feature simprints"
    )
    offsets: bool = Field(
        False, description="ISCC_SCT_OFFSETS - Include offsets of granular features"
    )
    chunks: bool = Field(False, description="ISCC_SCT_CHUNKS - Include granular text chunks")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ISCC_SCT_",
        extra="ignore",
        validate_assignment=True,
    )


sct_opts = SctOptions()
