import csv
import os
import time
from typing import Sequence, TextIO, Union

import numpy as np
import pandas as pd
import toolz
from absl import logging

from colosseum.utils.acme.base_logger import Logger, LoggingData
from colosseum.utils.acme.path import process_path


class CSVLogger(Logger):
    """
    A custom Logger class inspired by acme Logger.
    """

    _open = open

    @property
    def data(self) -> Sequence[LoggingData]:
        try:
            self.flush()
        except:
            pass
        data = pd.read_csv(self._file.name).to_dict()
        return toolz.valmap(lambda x: list(x.values()), data)

    def __init__(
        self,
        directory_or_file: Union[str, TextIO] = "tmp",
        label: str = "",
        time_delta: float = 0.0,
        add_uid: bool = True,
        flush_every: int = 30,
        file_name: str = "logs",
    ):
        os.makedirs(directory_or_file, exist_ok=True)

        self._label = label
        self._directory_or_file = directory_or_file
        self._file_name = file_name
        self._time_delta = time_delta
        self._flush_every = flush_every
        self._add_uid = add_uid

        self.reset()

        if flush_every <= 0:
            raise ValueError(
                f"`flush_every` must be a positive integer (got {flush_every})."
            )

    def _create_file(
        self,
        directory_or_file: Union[str, TextIO],
        label: str,
    ) -> TextIO:
        """Opens a file if input is a directory or use existing file."""
        if isinstance(directory_or_file, str):
            self._directory = process_path(
                directory_or_file, "logs", label, add_uid=self._add_uid
            )
            file_path = os.path.join(self._directory, f"{self._file_name}.csv")
            self._file_owner = True
            return self._open(file_path, mode="w")

        # TextIO instance.
        file = directory_or_file
        if label:
            logging.info("File, not directory, passed to CSVLogger; label not used.")
        if not file.mode.startswith("a"):
            raise ValueError(
                "File must be open in append mode; instead got " f'mode="{file.mode}".'
            )
        return file

    def write(self, data: LoggingData):
        """Writes a `data` into a row of comma-separated values."""
        # Only log if `time_delta` seconds have passed since last logging event.
        now = time.time()

        elapsed = now - self._last_log_time
        if elapsed < self._time_delta:
            logging.debug(
                "Not due to log for another %.2f seconds, dropping data.",
                self._time_delta - elapsed,
            )
            return
        self._last_log_time = now

        # Append row to CSV.
        data = toolz.valmap(np.array, data)
        # Use fields from initial `data` to create the header. If extra fields are
        # present in subsequent `data`, we ignore them.
        if not self._writer:
            fields = sorted(data.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=fields, extrasaction="ignore"
            )
            # Write header only if the file is empty.
            if not self._file.tell():
                self._writer.writeheader()
        self._writer.writerow(data)

        # Flush every `flush_every` writes.
        if self._writes % self._flush_every == 0:
            self.flush()
        self._writes += 1

    def close(self):
        self.flush()
        if self._file_owner:
            self._file.close()

    def flush(self):
        self._file.flush()

    def reset(self) -> None:
        self._last_log_time = time.time() - self._time_delta
        self._writer = None
        self._file_owner = False
        self._file = self._create_file(self._directory_or_file, self._label)
        self._writes = 0
        logging.info("Logging to %s", self.file_path)

    @property
    def file_path(self) -> str:
        return self._file.name
