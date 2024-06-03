from iscc_sct.models import SctMeta
from iscc_sct.code_semantic_text import gen_text_code_semantic
from iscc_sct.options import sct_opts


__all__ = [
    "create",
]


def create(text, granular=False, **options):
    # type (str, bool) -> SctMeta
    """
    Create Semantic Text-Code

    High-Level API for creating Semantic Text-Code.

    :param text: Text used for creating Semantic Text-Code.
    :param granular: Activate options for granular processing (Default: False).
    :param options: Override individual options for creating Semantic Text-Code.
    """

    # Override global options with individual options derived from `granular` parameter
    granular = dict(features=True, offsets=True, sizes=True, chunks=True) if granular else {}
    opts = sct_opts.override(granular)

    # Override local options with individual options form additional keyword arguments
    opts = opts.override(options)

    data = gen_text_code_semantic(text, **opts.model_dump())
    return SctMeta.from_dict(data)
