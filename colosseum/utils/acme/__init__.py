"""
This module is a partial fork of the [DeepMind acme](https://github.com/deepmind/acme/) package that allows to make use
of the loggers they have available without installing the package.
The reason why `acme` is not a dependency of the `Colosseum` package is because its installation has proven not to be
straightforward.
"""

from colosseum.utils.acme.csv_logger import CSVLogger
from colosseum.utils.acme.in_memory_logger import InMemoryLogger
