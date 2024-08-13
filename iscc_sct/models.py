"""
# Semantic-Code Text - Datamodel

This module provides the pydantic metadata schema for Semantic Text Code results.
The schema is conformant with https://schema.iscc.codes/

The `features` property of the top level Metadata Object support two different formats for
representing granular (per text chunk) features: the **Index-Format** and the **Object-Format**.
These formats are designed to offer flexibility in how feature data is structured and processed,
catering to different use cases where either performance or clarity is prioritized.

## Features Index-Format (Compact Array Structure):

In this compact format, features are represented as a list of strings, with optional parallel arrays to
store related attributes such as `offsets`, `sizes`, and `contents`.

**Example**:

```json
{
    "maintype": "semantic",
    "subtype": "text",
    "version": 0,
    "simprints": ["XZjeSfdyVi0", "NGrHC1F1Q-k"],
    "offsets": [0, 12],
    "sizes": [12, 48],
    "contents": ["textchunk no one", "textchunk no two"]
}

```

**Use Case**:
- Best suited for scenarios where storage efficiency is critical, and the overhead of processing
  multiple parallel arrays is acceptable.
- Useful when all features share the same set of attributes, allowing for faster bulk processing.

## Features Object-Format (Self-Descriptive Object Structure):

In this convenient format, each feature is represented as an individual object containing its
attributes (`feature`, `offset`, `size`, `content`). This makes the structure more verbose but
easier to read and work with.

**Example**:

```json
{
    "maintype": "content",
    "subtype": "text",
    "version": 0,
    "simprints": [
        {
            "simprint": "lUjuScFYBik",
            "offset": 0,
            "size": 25,
            "content": "ISCC - Semantic Text-Code"
        }
    ]
}

```
**Use Case**:
- Ideal for scenarios where clarity and readability are prioritized.
- Each feature is self-contained, making it easier to understand, extend, and debug.
- Flexibility in including or omitting optional attributes per feature.


### Unified FeatureSet Schema:

The `FeatureSet` model unifies these two formats by allowing either structure to be used.
To use the `FeatureSet` model, you can either provide data in the Index-Format or Object-Format.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel


__all__ = ["Feature", "FeatureSet", "Metadata"]


class PrettyBaseModel(BaseModel):
    def __repr__(self):
        return self.pretty_repr()

    def pretty_repr(self):
        return self.model_dump_json(indent=2, exclude_unset=True, exclude_none=True, exclude_defaults=False)


class Feature(PrettyBaseModel):
    simprint: str
    offset: Optional[int] = None
    size: Optional[int] = None
    content: Optional[str] = None


class FeatureSet(PrettyBaseModel):
    maintype: str = "semantic"
    subtype: str = "text"
    version: int = 0
    embedding: Optional[List[float]] = None
    simprints: Optional[
        Union[
            List[str],  # Index-Format
            List[Feature],  # Object-Format
        ]
    ] = None
    offsets: Optional[List[int]] = None
    sizes: Optional[List[int]] = None
    contents: Optional[List[str]] = None


class Metadata(PrettyBaseModel):
    iscc: str
    characters: Optional[int] = None
    features: Optional[List[FeatureSet]] = None
