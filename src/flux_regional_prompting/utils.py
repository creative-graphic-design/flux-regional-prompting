from typing import Tuple, TypedDict


class RegionalPromptMask(TypedDict):
    description: str
    mask: Tuple[int, int, int, int]
