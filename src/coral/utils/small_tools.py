"""
This module contains utility functions and classes for small tools.
"""

import os
import typing
from importlib.resources import files
from pathlib import Path

path = typing.Annotated[str | os.PathLike[str], "path"]


def resolve_package_path(path_to_file: str | Path) -> str:
    """
    Replace 'package://' at the start of the path with the chipiron package root.

    Args:
        path_to_file (str or Path): Input path, possibly starting with 'package://'.

    Returns:
        str: Resolved absolute path.
    """
    if isinstance(path_to_file, Path):
        path_to_file = str(path_to_file)

    if path_to_file.startswith("package://"):
        relative_path = path_to_file[len("package://") :]
        resource = files("chipiron").joinpath(relative_path)

        if not resource.is_file() and not resource.is_dir():
            raise FileNotFoundError(
                f"Resource not found: {relative_path} in package 'chipiron'"
            )

        return str(resource)  # You can also use `.as_posix()` if you need POSIX format
    return str(path_to_file)
