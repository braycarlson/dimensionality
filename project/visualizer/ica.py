from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import scienceplots
import seaborn as sns

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

    def biplot(self) -> Figure:
        plt.style.use('science')

        xs = self.transformation[:, 0]
        ys = self.transformation[:, 1]

        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())

        c = self.reduction.strategy.target
        figsize = self.engine.settings.figure.figsize
        scatter = self.engine.settings.scatter

        features = self.reduction.strategy.dataframe.columns.tolist()
        loadings = self.reduction.strategy.instance.components_

        if self.dimension == 3:
            projection = '3d'

            zs = self.transformation[:, 2]
            scale_z = 1.0 / (zs.max() - zs.min())

            transformation = (
                xs * scale_x,
                ys * scale_y,
                zs * scale_z
            )
        else:
            projection = None

            transformation = (
                xs * scale_x,
                ys * scale_y,
            )

        figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(111, projection=projection)

        ax.scatter(
            *transformation,
            c=c,
            **scatter.to_dict()
        )

        indices = range(
            1,
            self.reduction.strategy.instance.n_components + 1
        )

        index = [
            'Component ' + str(index)
            for index in indices
        ]

        padding = 1.08

        for i, feature in enumerate(features):
            if self.dimension == 3:
                ax.quiver(
                    0,
                    0,
                    0,
                    loadings[0, i],
                    loadings[1, i],
                    loadings[2, i],
                    color='r'
                )

                ax.text(
                    loadings[0, i] * padding,
                    loadings[1, i] * padding,
                    loadings[2, i] * padding,
                    feature,
                    ha='center',
                    va='center'
                )
            else:
                ax.arrow(
                    0,
                    0,
                    loadings[0, i],
                    loadings[1, i],
                    color='r',
                    head_length=0.03,
                    head_width=0.03
                )

                ax.text(
                    loadings[0, i] * padding,
                    loadings[1, i] * padding,
                    feature
                )

        plt.xlim(-1.10, 1.10)
        plt.ylim(-1.10, 1.10)

        xlabel = index[0]
        ylabel = index[1]

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Biplot for ICA')

        if self.dimension == 3:
            zlabel = index[2]
            ax.set_zlabel(zlabel)

        plt.tight_layout()

        self.engine.figure = figure

        return figure

    def heatmap(self) -> Figure:
        plt.style.use('science')

        indices = range(
            1,
            self.reduction.strategy.instance.components_.shape[0] + 1
        )

        index = [
            'Component ' + str(index)
            for index in indices
        ]

        columns = self.reduction.strategy.dataframe.columns.tolist()

        component = pd.DataFrame(
            self.reduction.strategy.instance.components_,
            columns=columns,
            index=index
        )

        figsize = self.engine.settings.figure.figsize
        figure = plt.figure(figsize=figsize)

        ax = sns.heatmap(
            component,
            annot=True,
            cmap='viridis'
        )

        ax.tick_params(
            axis='both',
            length=0,
            which='both'
        )

        plt.title('Heatmap for ICA')

        plt.tight_layout()

        self.engine.figure = figure

        return figure


class PlotlyVisualizer(BasePlotlyVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def biplot(self) -> go.Figure:
        color = self.reduction.strategy.target
        title = self.engine.settings.title

        xs = self.transformation[:, 0]
        ys = self.transformation[:, 1]

        scale_x = 1.0 / (xs.max() - xs.min())
        scale_y = 1.0 / (ys.max() - ys.min())

        if self.dimension == 3:
            instance = go.Scatter3d

            zs = self.transformation[:, 2]
            scale_z = 1.0 / (zs.max() - zs.min())

            scene = {
                'xaxis_title': 'Component 1',
                'yaxis_title': 'Component 2',
                'zaxis_title': 'Component 3'
            }

            transformation = {
                'x': xs * scale_x,
                'y': ys * scale_y,
                'z': zs * scale_z
            }
        else:
            instance = go.Scatter

            scene = {
                'xaxis_title': 'Component 1',
                'yaxis_title': 'Component 2',
            }

            transformation = {
                'x': xs * scale_x,
                'y': ys * scale_y
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

        features = self.reduction.strategy.dataframe.columns.tolist()
        components = self.reduction.strategy.instance.components_

        figure = go.Figure(data=data)
        figure.update_layout(scene=scene, title=title)

        if self.dimension == 3:
            for i, _ in enumerate(features):
                figure.add_trace(
                    go.Scatter3d(
                        x=[0, components[0, i]],
                        y=[0, components[1, i]],
                        z=[0, components[2, i]],
                        marker={'size': [0, 0]},
                        showlegend=False,
                        line={'width': 6}
                    )
                )
        else:
            for i, _ in enumerate(features):
                figure.add_trace(
                    go.Scatter(
                        x=[0, components[0, i]],
                        y=[0, components[1, i]],
                        marker={'size': [0, 0]},
                        showlegend=False,
                        line={'width': 2}
                    )
                )

        self.engine.figure = figure
        self.engine.method = 'biplot'

        return figure

    def heatmap(self) -> go.Figure:
        indices = range(
            1,
            self.reduction.strategy.instance.n_components + 1
        )

        index = [
            'Component ' + str(index)
            for index in indices
        ]

        columns = self.reduction.strategy.dataframe.columns.tolist()

        component = pd.DataFrame(
            self.reduction.strategy.instance.components_,
            columns=columns,
            index=index
        )

        figure = go.Figure(
            data=go.Heatmap(
                colorscale='Viridis',
                x=component.columns.tolist(),
                y=component.index.tolist(),
                z=component.to_numpy()
            )
        )

        figure.update_layout(
            title='Heatmap for ICA'
        )

        self.engine.figure = figure
        self.engine.method = 'heatmap'

        return figure


class ICAVisualizer:
    def __init__(
        self,
        visualizer: MatplotlibVisualizer | PlotlyVisualizer = None
    ):
        self.visualizer = visualizer

    def biplot(self) -> Figure:
        return self.visualizer.biplot()

    def heatmap(self) -> Figure:
        return self.visualizer.heatmap()

    def save(self, filename: str = None) -> None:
        return self.visualizer.save(filename)

    def show(self) -> None:
        return self.visualizer.show()

    def transform(self) -> Figure:
        return self.visualizer.transform()
