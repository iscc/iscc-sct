from iscc_sct.models import SctMeta


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
