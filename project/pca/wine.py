from __future__ import annotations

import pandas as pd

from project.reduction import (
    DimensionalityReduction,
    PCAStrategy
)
from project.visualizer.pca import (
    MatplotlibVisualizer,
    PCAVisualizer,
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

    strategy = PCAStrategy(dataframe=dataframe, dataset=dataset, target=target)
    reduction = DimensionalityReduction(strategy=strategy)

    n_components = 2
    transformation = reduction.reduce(n_components=n_components)

    visualizer = MatplotlibVisualizer(
        dimension=2,
        reduction=reduction,
        transformation=transformation
    )

    pca = PCAVisualizer(visualizer=visualizer)

    method = str(strategy)
    dimensionality = f"{n_components}-Component"

    title = f"{dimensionality} {method} for the Wine Dataset"
    pca.visualizer.engine.settings.title = title

    pca.transform()
    pca.show()


if __name__ == '__main__':
    main()
