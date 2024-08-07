import os
import os.path
import uuid
from typing import Optional, Tuple

# Pre-compute a unique identifier which is consistent within a single process.
_ACME_ID = uuid.uuid1()


def process_path(
    path: str,
    *subpaths: str,
    ttl_seconds: Optional[int] = None,
    backups: Optional[bool] = None,
    add_uid: bool = True
) -> str:
    """Process the path string.

    This will process the path string by running `os.path.expanduser` to replace
    any initial "~". It will also append a unique string on the end of the path
    and create the directories leading to this path if necessary.

    Args:
      path: string defining the path to process and create.
      *subpaths: potential subpaths to include after uniqification.
      ttl_seconds: ignored.
      backups: ignored.
      add_uid: Whether to add a unique directory identifier between `path` and
        `subpaths`. If FLAGS.acme_id is set, will use that as the identifier.

    Returns:
      the processed, expanded path string.
    """
    del backups, ttl_seconds

    path = os.path.expanduser(path)
    if add_uid:
        path = os.path.join(path, *get_unique_id())
    path = os.path.join(path, *subpaths)
    os.makedirs(path, exist_ok=True)
    return path


def get_unique_id() -> Tuple[str, ...]:
    """Makes a unique identifier for this process; override with FLAGS.acme_id."""
    # By default we'll use the global id.
    identifier = str(_ACME_ID)

    # Return as a tuple (for future proofing).
    return (identifier,)
