from __future__ import annotations

from pathlib import Path


def walk(file: Path) -> Path | None:
    for path in file.parents:
        if path.is_dir():
            venv = list(
                path.glob('venv')
            )

            for environment in venv:
                return environment.parent

            walk(path.parent)

    return None


file = Path.cwd()
CWD = walk(file).joinpath('project')

OUTPUT = CWD.joinpath('output')
OUTPUT.mkdir(exist_ok=True, parents=True)

SETTINGS = CWD.joinpath('settings')
SETTINGS.mkdir(exist_ok=True, parents=True)
