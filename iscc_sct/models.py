from typing import List, Optional, Dict, Any
from pydantic import BaseModel

__all__ = [
    "SctFeature",
    "SctMeta",
]


class SctFeature(BaseModel):
    feature: Optional[List[str]] = None
    offset: Optional[int] = None
    text: Optional[str] = None


class SctMeta(BaseModel):
    iscc: str
    characters: Optional[int] = None
    embedding: Optional[List[float]] = None
    features: Optional[List[SctFeature]] = None

    @classmethod
    def from_meta(cls, data: Dict[str, Any]) -> "SctMeta":
        # Initialize optional fields to None if they are not present in the data
        characters = data.get("characters")
        embedding = data.get("embedding")
        raw_features = data.get("features", [])

        # Convert features if present
        features = [
            SctFeature(feature=[f], offset=idx, text=chunk)
            for idx, (f, chunk) in enumerate(zip(raw_features, data.get("chunks", [])))
        ]

        return cls(
            iscc=data["iscc"],
            characters=characters,
            embedding=embedding,
            features=features if features else None,
        )
