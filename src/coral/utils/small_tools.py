"""This module contains utility functions and classes for small tools."""

import os
import typing
from importlib.resources import files
from pathlib import Path

import yaml

MyPath = typing.Annotated[str | os.PathLike[str], "path"]


class ResourceNotFoundError(FileNotFoundError):
    def __init__(self, relative_path: str, package: str) -> None:
        super().__init__(f"Resource not found: {relative_path} in package {package!r}")


def resolve_package_path(path_to_file: str | Path) -> str:
    """Replace 'package://' at the start of the path with the chipiron package root.

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
            raise ResourceNotFoundError(relative_path, "chipiron")

        return str(resource)  # You can also use `.as_posix()` if you need POSIX format
    return str(path_to_file)


def yaml_fetch_args_in_file(path_file: MyPath) -> dict[typing.Any, typing.Any]:
    """Fetch arguments from a YAML file.

    Args:
        path_file: The path to the YAML file.

    Returns:
        A dictionary containing the arguments.

    """
    with open(path_file, encoding="utf-8") as file:
        args: dict[typing.Any, typing.Any] = yaml.load(file, Loader=yaml.FullLoader)
    return args
