from __future__ import annotations

import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from abc import ABC, abstractmethod
from project.visualizer.engine import (
    MatplotlibEngine,
    PlotlyEngine
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.figure import Figure
    from project.reduction import DimensionalityReduction
    from project.visualizer.engine import Engine


class Visualizer(ABC):
    def __init__(
        self,
        dimension: int = 2,
        engine: Engine = None,
        reduction: DimensionalityReduction = None,
        transformation: npt.NDArray = None,
    ):
        self.dimension = dimension
        self.engine = engine
        self.reduction = reduction
        self.transformation = transformation

    def show(self) -> None:
        self.engine.show()

    def save(self, filename: str = None) -> None:
        self.engine.save(filename)

    @abstractmethod
    def transform(self) -> None:
        return NotImplementedError


class BaseMatplotlibVisualizer(Visualizer):
    def __init__(self, *args, **kwargs):
        engine = MatplotlibEngine()

        super().__init__(*args, **kwargs, engine=engine)

    def __repr__(self) -> str:
        return 'matplotlib'

    def __str__(self) -> str:
        return 'matplotlib'

    def blender(self, column: list[str] = None) -> None:
        plt.style.use('science')

        if column is None:
            column = self.reduction.strategy.dataframe.columns

        combinations = itertools.combinations(column, 2)

        figures = []

        for combination in combinations:
            x, y = combination

            dataset = (
                self.reduction.strategy.dataframe[x],
                self.reduction.strategy.dataframe[y]
            )

            c = self.reduction.strategy.target
            figsize = self.engine.settings.figure.figsize
            scatter = self.engine.settings.scatter

            figure = plt.figure(figsize=figsize)
            ax = figure.add_subplot(111)

            ax.scatter(
                *dataset,
                c=c,
                **scatter.to_dict()
            )

            plt.xlabel(x)
            plt.ylabel(y)

            title = f"{x} vs. {y}"
            plt.title(title)

            figures.append(figure)

        self.engine.figure = figures
        self.engine.method = 'blender'

        return figures

    def transform(self) -> Figure:
        plt.style.use('science')

        c = self.reduction.strategy.target
        figsize = self.engine.settings.figure.figsize
        scatter = self.engine.settings.scatter
        title = self.engine.settings.title

        if self.dimension == 3:
            projection = '3d'

            transformation = (
                self.transformation[:, 0],
                self.transformation[:, 1],
                self.transformation[:, 2]
            )
        else:
            projection = None

            transformation = (
                self.transformation[:, 0],
                self.transformation[:, 1]
            )

        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111, projection=projection)

        ax.scatter(
            *transformation,
            c=c,
            **scatter.to_dict()
        )

        plt.title(title)

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        if self.dimension == 3:
            plt.ylabel('Component 3')

        self.engine.figure = figure

        return figure


class BasePlotlyVisualizer(Visualizer):
    def __init__(self, *args, **kwargs):
        engine = PlotlyEngine()

        super().__init__(*args, **kwargs, engine=engine)

    def __repr__(self) -> str:
        return 'plotly'

    def __str__(self) -> str:
        return 'plotly'

    def blender(self, column: list[str] = None) -> None:
        if column is None:
            column = self.reduction.strategy.dataframe.columns

        combinations = itertools.combinations(column, 2)

        figures = []

        for combination in combinations:
            x, y = combination

            dataset = {
                'x': self.reduction.strategy.dataframe[x],
                'y': self.reduction.strategy.dataframe[y]
            }

            color = self.reduction.strategy.target

            marker = {
                'size': 6,
                'color': color,
                'colorscale': 'Viridis',
            }

            title = f"{x} vs. {y}"

            data = go.Scatter(
                mode='markers',
                marker=marker,
                **dataset
            )

            figure = go.Figure(data=data)

            figure.update_layout(
                title=title,
                xaxis_title=x,
                yaxis_title=y
            )

            figures.append(figure)

        self.engine.figure = figures
        self.engine.method = 'blender'

        return figures

    def transform(self) -> go.Figure:
        color = self.reduction.strategy.target
        title = self.engine.settings.title

        if self.dimension == 3:
            instance = go.Scatter3d

            scene = {
                'xaxis_title': 'Component 1',
                'yaxis_title': 'Component 2',
                'zaxis_title': 'Component 3'
            }

            transformation = {
                'x': self.transformation[:, 0],
                'y': self.transformation[:, 1],
                'z': self.transformation[:, 2]
            }
        else:
            instance = go.Scatter

            scene = {
                'xaxis_title': 'Component 1',
                'yaxis_title': 'Component 2'
            }

            transformation = {
                'x': self.transformation[:, 0],
                'y': self.transformation[:, 1]
            }

        data = instance(
            **transformation,
            mode='markers',
            marker={
                'size': 6,
                'color': color,
                'colorscale': 'Viridis',
            }
        )

        figure = go.Figure(data=data)
        figure.update_layout(scene=scene, title=title)

        self.engine.figure = figure
        self.engine.method = 'transform'

        return figure
