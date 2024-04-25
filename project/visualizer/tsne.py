from __future__ import annotations

import scienceplots

from project.visualizer.visualizer import (
    BaseMatplotlibVisualizer,
    BasePlotlyVisualizer,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class MatplotlibVisualizer(BaseMatplotlibVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlotlyVisualizer(BasePlotlyVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TSNEVisualizer:
    def __init__(
        self,
        visualizer: MatplotlibVisualizer | PlotlyVisualizer = None
    ):
        self.visualizer = visualizer

    def save(self, filename: str = None) -> None:
        return self.visualizer.save(filename)

    def show(self) -> None:
        return self.visualizer.show()

    def transform(self) -> Figure:
        return self.visualizer.transform()
