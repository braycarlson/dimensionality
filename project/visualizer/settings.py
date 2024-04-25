from __future__ import annotations

import json

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    from typing_extensions import Any, Self


class Settings:
    def __init__(self, **data: Any):
        for k, v in data.items():
            if isinstance(v, Mapping):
                self.__dict__[k] = Settings(**v)
            else:
                self.__dict__[k] = v

    def __getattr__(self, name: str):
        if name not in self.__dict__:
            self.__dict__[name] = Settings()

        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Mapping):
            self.__dict__[name] = Settings(**value)
        else:
            self.__dict__[name] = value

    def __repr__(self):
        return repr(self.__dict__)

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        return cls(**data)

    @classmethod
    def from_file(cls: type[Self], path: str | pathlib.Path) -> Self:
        try:
            with open(path, 'r') as handle:
                data = json.load(handle)
                return cls(**data)
        except Exception as exceptione:
            message = 'Error loading from file.'
            raise OSError(message) from exceptione

    @classmethod
    def from_object(cls: type[Self], data: object) -> Self:
        return cls(**data.__dict__)

    def update(self, data: dict[str, Any]) -> None:
        for k, v in data.items():
            if isinstance(v, Mapping) and isinstance(self.__dict__.get(k), Mapping):
                self.__dict__[k].update(v)
            else:
                self.__dict__[k] = v

    def save(self, path: str | pathlib.Path) -> None:
        try:
            with open(path, 'w+') as file:
                json.dump(
                    self.__dict__,
                    file,
                    indent=4
                )
        except Exception as exception:
            message = 'Error saving to file.'
            raise OSError(message) from exception

    def to_dict(self) -> dict[Any, Any]:
        result = {}

        for k, v in self.__dict__.items():
            if isinstance(v, Settings):
                result[k] = v.to_dict()
            else:
                result[k] = v

        return result
