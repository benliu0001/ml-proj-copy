from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import csv
import random

QuestionIndex = int
UserIndex = int


@dataclass
class IRTDataPoint:
    question: QuestionIndex
    user: UserIndex
    correct: int


class IRTDataset:
    _data: list[IRTDataPoint]

    def __init__(self, data: list[IRTDataPoint]) -> None:
        """
        Do not call this constructor directly. Use `IRTDataset.from_data` or
        `IRTDataset.from_file` instead.
        """

        self._data = data

    @classmethod
    def from_data(cls, data: list[IRTDataPoint]) -> IRTDataset:
        return cls(data)

    @classmethod
    def from_file(cls, path: str) -> IRTDataset:
        data = []

        with open(path) as fl:
            reader = csv.reader(fl)

            # Skip header
            next(reader)

            for row in reader:
                if len(row) != 3:
                    continue

                data.append(
                    IRTDataPoint(
                        question=int(row[0]),
                        user=int(row[1]),
                        correct=int(row[2]),
                    )
                )
        
        return cls(data)

    def iter(self) -> Iterable[IRTDataPoint]:
        return (d for d in self._data)
    
    def __len__(self) -> int:
        return len(self._data)

    def n_questions(self) -> int:
        return max(d.question for d in self._data) + 1

    def n_users(self) -> int:
        return max(d.user for d in self._data) + 1

    def extend(self, other: IRTDataset) -> None:
        self._data.extend(other._data)

    def shuffle(self) -> None:
        random.shuffle(self._data)

    def batch(self, batch_size: int) -> Iterable[IRTDataset]:
        for i in range(0, len(self), batch_size):
            yield IRTDataset.from_data(self._data[i:i + batch_size])