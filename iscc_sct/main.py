from iscc_sct.models import Metadata
from iscc_sct.code_semantic_text import gen_text_code_semantic
from iscc_sct.options import sct_opts


__all__ = [
    "create",
]


def create(text, granular=False, **options):
    # type: (str, bool) -> Metadata
    """
    Create Semantic Text-Code

    High-Level API for creating Semantic Text-Code.

    :param text: Text used for creating Semantic Text-Code.
    :param granular: Activate options for granular processing (Default: False).
    :param options: Override individual options for creating Semantic Text-Code.
    :key model_version (int): Model version to use (0=minilm-l12, 1=embeddinggemma-300m, default from environment or 0).
    :key bits (int): Length of generated Semantic Text-Code in bits (default 64).
    :key bits_granular (int): Bit-length of granular features (default 64).
    :key max_tokens (int): Max tokens per chunk (default 127, max 2048).
    :key overlap (int): Max tokens allowed overlapping between chunks (default 48).
    :return: Semantic Text-Code `Metadata` object in Object-Format
    """

    # Override global options with individual options derived from `granular` parameter
    granular_opts = (
        dict(simprints=True, offsets=True, sizes=True, contents=True) if granular else {}
    )
    opts = sct_opts.override(granular_opts)

    # Override local options with individual options from additional keyword arguments
    opts = opts.override(options)

    data = gen_text_code_semantic(text, **opts.model_dump())
    return Metadata(**data).to_object_format()
