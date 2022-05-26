import abc
from typing import Any, Mapping, Sequence

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    """A logger has a `write` method."""

    @property
    @abc.abstractmethod
    def data(self) -> Sequence[LoggingData]:
        pass

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Empties the logger."""


class NoOpLogger(Logger):
    """Simple Logger which does nothing and outputs no logs.

    This should be used sparingly, but it can prove useful if we want to quiet an
    individual component and have it produce no logging whatsoever.
    """

    @property
    def data(self) -> Sequence[LoggingData]:
        raise NotImplementedError()

    def write(self, data: LoggingData):
        pass

    def close(self):
        pass
