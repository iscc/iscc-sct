"""
# Semantic-Code Text - Datamodel

This module provides the pydantic metadata schema for Semantic Text Code results.
The schema is conformant with https://schema.iscc.codes/

The `features` property of the top level Metadata Object supports two different formats for
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

from typing import List, Optional, Union
from pydantic import BaseModel


__all__ = ["Feature", "FeatureSet", "Metadata"]


class PrettyBaseModel(BaseModel):
    def __repr__(self):
        return self.pretty_repr()

    def pretty_repr(self):
        return self.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True, exclude_defaults=False
        )


class Feature(PrettyBaseModel):
    simprint: str
    offset: Optional[int] = None
    size: Optional[int] = None
    content: Optional[str] = None


class FeatureSet(PrettyBaseModel):
    maintype: str = "semantic"
    subtype: str = "text"
    version: int = 0
    byte_offsets: Optional[bool] = False
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

    def to_index_format(self) -> "Metadata":
        """
        Convert the Metadata object to use the Index-Format for features.
        Returns a new Metadata object.
        """
        if not self.features:
            return self.model_copy()

        new_features = []
        for feature_set in self.features:
            new_feature_set = feature_set.model_copy()
            if feature_set.simprints is None:
                new_features.append(new_feature_set)
                continue

            if isinstance(feature_set.simprints[0], str):
                new_features.append(new_feature_set)
            else:
                new_feature_set.simprints = [f.simprint for f in feature_set.simprints]
                new_feature_set.offsets = [
                    f.offset for f in feature_set.simprints if f.offset is not None
                ]
                new_feature_set.sizes = [
                    f.size for f in feature_set.simprints if f.size is not None
                ]
                new_feature_set.contents = [
                    f.content for f in feature_set.simprints if f.content is not None
                ]
                new_features.append(new_feature_set)

        return Metadata(iscc=self.iscc, characters=self.characters, features=new_features)

    def get_content(self) -> Optional[str]:
        """
        Reconstruct and return the original input text if all necessary data is available.
        This method removes overlaps in adjacent text chunks.

        :return: The reconstructed original text, or None if the necessary data is not available.
        """
        if not self.features or not self.features[0].simprints:
            return None

        feature_set = self.features[0]
        if isinstance(feature_set.simprints[0], str):
            # Convert to object format if in index format
            feature_set = self.to_object_format().features[0]

        if not all(
            feature.content and feature.offset is not None for feature in feature_set.simprints
        ):
            return None

        # Sort features by offset
        sorted_features = sorted(feature_set.simprints, key=lambda x: x.offset)

        reconstructed_text = ""
        last_end = 0

        for feature in sorted_features:
            start = feature.offset
            if start < last_end:
                # Remove overlap
                feature_content = feature.content[last_end - start :]
            else:
                feature_content = feature.content

            reconstructed_text += feature_content
            last_end = start + len(feature.content)

        return reconstructed_text

    def get_overlaps(self) -> List[str]:
        """
        Returns a list of overlapping text between consecutive chunks.
        For non-overlapping consecutive chunks, returns an empty string.

        :return: List of overlapping text or empty strings.
        """
        if not self.features or not self.features[0].simprints:
            return []

        feature_set = self.features[0]
        if isinstance(feature_set.simprints[0], str):
            # Convert to object format if in index format
            feature_set = self.to_object_format().features[0]

        if not all(
            feature.content and feature.offset is not None for feature in feature_set.simprints
        ):
            return []

        # Sort features by offset
        sorted_features = sorted(feature_set.simprints, key=lambda x: x.offset)
        overlaps = []

        for i in range(len(sorted_features) - 1):
            current_feature = sorted_features[i]
            next_feature = sorted_features[i + 1]

            current_end = current_feature.offset + len(current_feature.content)
            next_start = next_feature.offset

            if current_end > next_start:
                overlap = current_feature.content[next_start - current_feature.offset :]
                overlaps.append(overlap)
            else:
                overlaps.append("")

        return overlaps

    def to_object_format(self) -> "Metadata":
        """
        Convert the Metadata object to use the Object-Format for features.
        Returns a new Metadata object.
        """
        if not self.features:
            return self.model_copy()

        new_features = []
        for feature_set in self.features:
            new_feature_set = feature_set.model_copy()
            if feature_set.simprints is None:
                new_features.append(new_feature_set)
                continue

            if isinstance(feature_set.simprints[0], Feature):
                new_features.append(new_feature_set)
            else:
                new_simprints = []
                for i, simprint in enumerate(feature_set.simprints):
                    feature = Feature(simprint=simprint)
                    if feature_set.offsets and i < len(feature_set.offsets):
                        feature.offset = feature_set.offsets[i]
                    if feature_set.sizes and i < len(feature_set.sizes):
                        feature.size = feature_set.sizes[i]
                    if feature_set.contents and i < len(feature_set.contents):
                        feature.content = feature_set.contents[i]
                    new_simprints.append(feature)
                new_feature_set.simprints = new_simprints
                new_feature_set.offsets = None
                new_feature_set.sizes = None
                new_feature_set.contents = None
                new_features.append(new_feature_set)

        return Metadata(iscc=self.iscc, characters=self.characters, features=new_features)
