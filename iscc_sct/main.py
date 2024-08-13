from iscc_sct.models import Metadata
from iscc_sct.code_semantic_text import gen_text_code_semantic
from iscc_sct.options import sct_opts


__all__ = [
    "create",
]


def create(text, granular=False, **options):
    # type (str, bool) -> Metadata
    """
    Create Semantic Text-Code

    High-Level API for creating Semantic Text-Code.

    :param text: Text used for creating Semantic Text-Code.
    :param granular: Activate options for granular processing (Default: False).
    :param options: Override individual options for creating Semantic Text-Code.
    :return: Semantic Text-Code `Metadata` object in Object-Format
    """

    # Override global options with individual options derived from `granular` parameter
    granular = dict(simprints=True, offsets=True, sizes=True, contents=True) if granular else {}
    opts = sct_opts.override(granular)

    # Override local options with individual options form additional keyword arguments
    opts = opts.override(options)

    data = gen_text_code_semantic(text, **opts.model_dump())
    return Metadata(**data).to_object_format()
