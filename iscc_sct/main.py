from iscc_sct.models import SctMeta
from iscc_sct.code_semantic_text import gen_text_code_semantic


__all__ = [
    "create",
]


def create(text, granular=False, **options):
    # type (str, bool) -> SctMeta
    """
    Create Semantic Text-Code

    High-Level API for creating Semantic Text-Code.

    :param text: Text used for creating Semantic Text-Code.
    :param granular: Return detailed granular (per chunk) features.
    :param options: Additional options for processing.
    """
def create(text, granular=False, **options):
    # type (str, bool) -> SctMeta
    """
    Create Semantic Text-Code

    High-Level API for creating Semantic Text-Code.

    :param text: Text used for creating Semantic Text-Code.
    :param granular: Return detailed granular (per chunk) features.
    :param options: Additional options for processing.
    """
    options['features'] = granular
    result = gen_text_code_semantic(text, **options)
    return SctMeta.from_meta(result)
