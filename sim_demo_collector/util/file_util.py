import os
from pathlib import Path
from typing import Optional, Union


def str2path(path: str) -> Path:
    return Path(os.path.expanduser(path)).absolute()


def mkdir(
    path: Union[str, Path],
    parents: Optional[bool] = False,
    exist_ok: Optional[bool] = False
) -> Path:
    path = str2path(path)
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path