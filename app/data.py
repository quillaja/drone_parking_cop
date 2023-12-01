import re
from typing import Callable

from frame.processor import Plate

PlateComparer = Callable[[Plate, Plate], Plate]
"""
A function that determines which plate is "better" and returns it.
PlateComparer(old, new) -> bool
"""

PLATE_NUMBER = re.compile(r"[A-Z0-9]{4,7}")
"""regexp to id a variety of plates"""


def default_comparer(old: Plate, new: Plate) -> Plate:
    """Keep new if text is 4-7 alphanumeric and new confidence > old confidence."""
    isValidNumber = PLATE_NUMBER.match(new.text)
    if isValidNumber and new.confidence > old.confidence:
        return new
    else:
        return old


class ResultAccumulator:
    def __init__(self, comparer: PlateComparer = default_comparer) -> None:
        self.comparer = comparer
        self.frames: list[list[Plate]] = []
        self.best: dict[int, Plate] = {}

    @property
    def total_found(self) -> int:
        return len(self.best)

    def add(self, results: list[Plate]):
        """
        Add newly detected results.
        """
        self.frames.append(results)
        self._update_max()

    def _update_max(self):
        """
        Updates each tracked plate using the highest the most recent list of plates.
        """
        for new_plate in self.frames[-1]:
            id = new_plate.track_id
            prev_plate = self.best.setdefault(id, new_plate)
            self.best[id] = self.comparer(prev_plate, new_plate)
