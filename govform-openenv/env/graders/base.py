"""Abstract base grader for task scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseGrader(ABC):
    """Every task-specific grader must implement ``grade``."""

    @abstractmethod
    def grade(self, final_state: Dict[str, Any]) -> float:
        """
        Return a score in [0.0, 1.0] given the final environment state dict.
        """
        ...
