from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class GenericRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    row: Dict[str, Any]
    row_idx: int
    metadata: Dict[str, Any]
    response_format: Optional[Type[BaseModel]] = None
