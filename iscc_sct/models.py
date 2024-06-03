from typing import List, Optional, Dict, Any
from pydantic import BaseModel

__all__ = [
    "SctFeature",
    "SctMeta",
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

