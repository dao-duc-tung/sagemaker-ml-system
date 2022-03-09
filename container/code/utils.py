import logging
import os
import sys
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def list_dir(path: Union[str, Path]):
    files = []
    path = Path(path)
    if not path.exists():
        logger.warn(f"{path.as_posix()} doesn't exist.")
    else:
        files = os.listdir(str(path))
    return files
