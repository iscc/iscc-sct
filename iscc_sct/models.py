from typing import List, Optional, Dict, Any
from pydantic import BaseModel

__all__ = [
    "SctFeature",
    "SctMeta",
]
]

class SctMeta(BaseModel):
    iscc: str
    characters: Optional[int] = None
    embedding: Optional[List[float]] = None
    features: Optional[List[SctFeature]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SctMeta":
        features = None
        if "features" in data:
            features = [SctFeature(feature=f) for f in data["features"]]
        return cls(
            iscc=data["iscc"],
            characters=data.get("characters"),
            embedding=data.get("embedding"),
            features=features,
        )
]


class SctFeature(BaseModel):
    feature: Optional[str] = None
    offset: Optional[int] = None
    size: Optional[int] = None
    text: Optional[str] = None


class SctMeta(BaseModel):
    iscc: str
    characters: Optional[int] = None
    embedding: Optional[List[float]] = None
    features: Optional[List[SctFeature]] = None

