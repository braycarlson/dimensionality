from __future__ import annotations

import pandas as pd

from project.reduction import (
    DimensionalityReduction,
    ICAStrategy,
    PCAStrategy,
    TSNEStrategy,
    UMAPStrategy
)
from project.visualizer.pca import (
    MatplotlibVisualizer,
    PCAVisualizer,
    PlotlyVisualizer
)
from project.visualizer.settings import Settings
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine
)
from sklearn.preprocessing import StandardScaler


def main() -> None:
    scale = True

    loader = load_iris()

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

    dimensionality = f"{n_components}-Component"
    method = str(strategy)

    title = f"{dimensionality} {method} for the Iris Dataset"
    pca.visualizer.engine.settings.title = title

    dimensionality = dimensionality.lower()
    method = method.lower()
    visualizer = str(visualizer)

    filename = f"{method}_{visualizer}_{dimensionality}_blender.png"

    pca.blender()
    # pca.show()
    pca.save(filename=filename)

    filename = f"{method}_{visualizer}_{dimensionality}_biplot.png"

    pca.biplot()
    # pca.show()
    pca.save(filename=filename)

    filename = f"{method}_{visualizer}_{dimensionality}_explain.png"

    pca.explain()
    # pca.show()
    pca.save(filename=filename)

    filename = f"{method}_{visualizer}_{dimensionality}_heatmap.png"

    pca.heatmap()
    # pca.show()
    pca.save(filename=filename)

    filename = f"{method}_{visualizer}_{dimensionality}_ratio.png"

    pca.ratio()
    # pca.show()
    pca.save(filename=filename)

    filename = f"{method}_{visualizer}_{dimensionality}_transform.png"

    pca.transform()
    # pca.show()
    pca.save(filename=filename)


if __name__ == '__main__':
    main()
