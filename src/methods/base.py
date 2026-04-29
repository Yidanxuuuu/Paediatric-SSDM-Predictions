from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class MethodResult:
    method: str
    outputs: dict[str, Any]


class MethodRunner(Protocol):
    def run(self, *args, **kwargs) -> MethodResult:
        ...

