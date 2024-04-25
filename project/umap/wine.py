from __future__ import annotations

import pandas as pd

from project.reduction import (
    DimensionalityReduction,
    UMAPStrategy
)
from project.visualizer.umap import (
    MatplotlibVisualizer,
    UMAPVisualizer,
    PlotlyVisualizer
)
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def main() -> None:
    scale = True

    loader = load_wine()

    columns, data, target = (
        loader.feature_names,
        loader.data,
        loader.target
    )

    dataframe = pd.DataFrame(columns=columns, data=data)

    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(dataframe)

        dataframe = pd.DataFrame(columns=columns, data=data)

    dataset = dataframe.to_numpy()

    strategy = UMAPStrategy(dataframe=dataframe, dataset=dataset, target=target)
    reduction = DimensionalityReduction(strategy=strategy)

    n_components = 2
    transformation = reduction.reduce(n_components=n_components)

    visualizer = MatplotlibVisualizer(
        dimension=2,
        reduction=reduction,
        transformation=transformation
    )

    umap = UMAPVisualizer(visualizer=visualizer)

    method = str(strategy)
    dimensionality = f"{n_components}-Component"

    title = f"{dimensionality} {method} for the Wine Dataset"
    umap.visualizer.engine.settings.title = title

    umap.transform()
    umap.show()


if __name__ == '__main__':
    main()
