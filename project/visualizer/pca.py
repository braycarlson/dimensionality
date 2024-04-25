from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scienceplots
import seaborn as sns

from bioinfokit.visuz import cluster
from matplotlib import ticker
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
        plt.title('Biplot for PCA')

        if self.dimension == 3:
            zlabel = index[2]
            ax.set_zlabel(zlabel)

        plt.tight_layout()

        self.engine.figure = figure

        return figure

    def explain(self) -> Figure:
        plt.style.use('science')

        figsize = self.engine.settings.figure.figsize
        figure = plt.figure(figsize=figsize)

        variance = self.reduction.strategy.instance.explained_variance_

        plt.bar(
            range(1, len(variance) + 1),
            variance
        )

        plt.plot(
            range(1, len(variance) + 1),
            np.cumsum(variance),
            c='red',
            label='Cumulative Explained Variance'
        )

        null = plt.NullLocator()
        multiple = ticker.MultipleLocator(1)

        plt.gca().xaxis.set_minor_locator(null)
        plt.gca().xaxis.set_major_locator(multiple)

        plt.ylabel('Explained Variance')
        plt.xlabel('Component')

        plt.legend(loc='upper left')

        plt.title('Explained Variance Plot for PCA')

        plt.tight_layout()

        self.engine.figure = figure

        return figure

    def heatmap(self) -> Figure:
        plt.style.use('science')

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

        plt.title('Heatmap for PCA')

        plt.tight_layout()

        self.engine.figure = figure

        return figure

    def ratio(self) -> Figure:
        figsize = self.engine.settings.figure.figsize
        figure = plt.figure(figsize=figsize)

        plt.plot(self.reduction.strategy.instance.explained_variance_ratio_)

        null = plt.NullLocator()
        multiple = ticker.MultipleLocator(1)

        ax = plt.gca()

        ax.xaxis.set_minor_locator(null)
        ax.xaxis.set_major_locator(multiple)

        x = ax.get_xticks().astype(int)

        label = x + 1

        locator = ticker.FixedLocator(x)
        ax.xaxis.set_major_locator(locator)

        formatter = ticker.FixedFormatter(label)
        ax.xaxis.set_major_formatter(formatter)

        plt.xlabel('Component')
        plt.ylabel('Cumulative Explained Variance')

        plt.title('Explained Variance Ratio Plot for PCA')

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

    def explain(self) -> go.Figure:
        variance = self.reduction.strategy.instance.explained_variance_
        cumulative = np.cumsum(variance)

        x = list(
            range(1, len(variance) + 1)
        )

        figure = go.Figure(
            data=[
                go.Bar(
                    name='Explained Variance',
                    x=x,
                    y=variance,
                ),
                go.Scatter(
                    name='Cumulative Explained Variance',
                    x=x,
                    y=cumulative,
                )
            ]
        )

        figure.update_layout(
            xaxis = {
                'dtick': 1
            },
            xaxis_title='Component',
            yaxis_title='Variance',
            title='Explained Variance Plot for PCA'
        )

        self.engine.figure = figure
        self.engine.method = 'explain'

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

        index = component.index.tolist()
        columns = component.columns.tolist()
        z = component.to_numpy()

        for i in range(len(index)):
            for j in range(len(columns)):
                amount = z[i][j]
                amount = round(amount, 2)
                text = str(amount)

                figure.add_annotation(
                    {
                        'x': j,
                        'y': i,
                        'text': text,
                        'showarrow': False,
                        'font': {
                            'size': 12,
                            'color': 'black'
                        }
                    }
                )

        figure.update_layout(
            title='Heatmap for PCA'
        )

        self.engine.figure = figure
        self.engine.method = 'heatmap'

        return figure

    def ratio(self) -> go.Figure:
        ratio = self.reduction.strategy.instance.explained_variance_ratio_

        x = list(
            range(1, len(ratio) + 1)
        )

        figure = go.Figure(
            data=go.Scatter(
                x=x,
                y=ratio,
                mode='lines'
            )
        )

        figure.update_layout(
            xaxis = {
                'dtick': 1
            },
            xaxis_title='Component',
            yaxis_title='Cumulative Explained Variance',
            title='Explained Variance Ratio Plot for PCA'
        )

        self.engine.figure = figure
        self.engine.method = 'ratio'

        return figure


class PCAVisualizer:
    def __init__(
        self,
        visualizer: MatplotlibVisualizer | PlotlyVisualizer = None
    ):
        self.visualizer = visualizer

    def biplot(self) -> Figure:
        return self.visualizer.biplot()

    def blender(self) -> Figure:
        return self.visualizer.blender()

    def explain(self) -> Figure:
        return self.visualizer.explain()

    def heatmap(self) -> Figure:
        return self.visualizer.heatmap()

    def ratio(self) -> Figure:
        return self.visualizer.ratio()

    def save(self, filename: str = None) -> None:
        return self.visualizer.save(filename)

    def show(self) -> None:
        return self.visualizer.show()

    def transform(self) -> Figure:
        return self.visualizer.transform()
