from __future__ import annotations

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from plotly import offline
from project.constant import OUTPUT, SETTINGS
from project.visualizer.settings import Settings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class Engine(ABC):
    def __init__(self, settings: dict[Any, Any] = None):
        self.figure = None
        self.method = None
        self.settings = settings

    @abstractmethod
    def save(self, filename: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError


class MatplotlibEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = SETTINGS.joinpath('matplotlib.json')
        self.settings = Settings.from_file(path)

    def __repr__(self) -> str:
        return 'matplotlib'

    def __str__(self) -> str:
        return 'matplotlib'

    def save(self, filename: str) -> None:
        if isinstance(self.figure, list):
            for index, figure in enumerate(self.figure, 0):
                name = f"{index}_{filename}"
                path = OUTPUT.joinpath(name)

                figure.savefig(
                    path,
                    bbox_inches='tight',
                    dpi=300,
                    format='png'
                )

                plt.close()
        else:
            path = OUTPUT.joinpath(filename)

            self.figure.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png'
            )

            plt.close()

    def show(self) -> None:
        if isinstance(self.figure, list):
            for figure in self.figure:
                plt.figure(figure.number)
                plt.show()
                plt.close()
        else:
            plt.figure(self.figure.number)
            plt.show()
            plt.close()


class PlotlyEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = SETTINGS.joinpath('matplotlib.json')
        self.settings = Settings.from_file(path)

    def __repr__(self) -> str:
        return 'plotly'

    def __str__(self) -> str:
        return 'plotly'

    def save(self, filename: str) -> None:
        if isinstance(self.figure, list):
            for index, figure in enumerate(self.figure, 0):
                name = f"{index}_{filename}"
                path = OUTPUT.joinpath(name)

                figure.write_image(path, engine='kaleido')
        else:
            path = OUTPUT.joinpath(filename)

            self.figure.write_image(path, engine='kaleido')

    def show(self) -> None:
        if isinstance(self.figure, list):
            for index, figure in enumerate(self.figure, 0):
                filename = f"{index}_plotly_{self.method}.html"
                filename = OUTPUT.joinpath(filename).as_posix()

                offline.plot(figure, filename=filename)
        else:
            filename = f"plotly_{self.method}.html"
            filename = OUTPUT.joinpath(filename).as_posix()

            offline.plot(self.figure, filename=filename)
