from typing import List, Optional, Dict, Any
from pydantic import BaseModel

__all__ = ["SctFeature", "SctMeta"]


class PrettyBaseModel(BaseModel):
    def __repr__(self):
        return self.pretty_repr()

    def pretty_repr(self):
        return self.model_dump_json(exclude_unset=True, exclude_none=True, indent=2)


class SctFeature(PrettyBaseModel):
    feature: Optional[str] = None
    offset: Optional[int] = None
    size: Optional[int] = None
    text: Optional[str] = None


class SctMeta(PrettyBaseModel):
    iscc: str
    characters: Optional[int] = None
    embedding: Optional[List[float]] = None
    features: Optional[List[SctFeature]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SctMeta":
        features = []
        feature_list = data.get("features", [])
        offset_list = data.get("offsets", [])
        size_list = data.get("sizes", [])
        text_list = data.get("chunks", [])

        max_len = max(len(feature_list), len(offset_list), len(size_list), len(text_list))

        for i in range(max_len):
            features.append(
                SctFeature(
                    feature=feature_list[i] if feature_list else None,
                    offset=offset_list[i] if offset_list else None,
                    size=size_list[i] if size_list else None,
                    text=text_list[i] if text_list else None,
                )
            )
        return cls(
            iscc=data["iscc"],
            characters=data.get("characters"),
            embedding=data.get("embedding"),
            features=features if features else None,
        )
