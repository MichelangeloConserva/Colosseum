from typing import Sequence

from colosseum.utils.acme.base_logger import Logger, LoggingData


class InMemoryLogger(Logger):
    """A simple logger that keeps all data in memory."""

    def __init__(self):
        self.reset()

    def write(self, data: LoggingData):
        self._data.append(data)

    def close(self):
        pass

    def reset(self) -> None:
        self._data = []

    @property
    def data(self) -> Sequence[LoggingData]:
        return self._data
