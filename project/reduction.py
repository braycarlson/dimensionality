from __future__ import annotations

import warnings

from numba import NumbaDeprecationWarning

warnings.filterwarnings(
    'ignore',
    category=NumbaDeprecationWarning
)

from abc import ABC, abstractmethod
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from typing import TYPE_CHECKING
from umap import UMAP

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from typing_extensions import Any


class DimensionalityReduction:
    def __init__(self, strategy: Any = None):
        self.strategy = strategy

    def heatmap(self) -> None:
        self.strategy.heatmap()

    def reduce(self, *args, **kwargs) -> None:
        return self.strategy.reduce(*args, **kwargs)

    def scree(self) -> None:
        self.strategy.scree()

    def variance(self) -> None:
        self.strategy.variance()


class DimensionalityReductionStrategy(ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        dataset: npt.NDArray = None,
        target: npt.NDArray = None
    ):
        self.dataframe = dataframe
        self.dataset = dataset
        self.instance = None
        self.target = target

    @abstractmethod
    def reduce(self, *args, **kwargs) -> None:
        raise NotImplementedError


class ICAStrategy(DimensionalityReductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return 'ICA'

    def __str__(self) -> str:
        return 'ICA'

    def reduce(self, *args, **kwargs) -> None:
        self.instance = FastICA(*args, **kwargs)
        return self.instance.fit_transform(self.dataset)


class PCAStrategy(DimensionalityReductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return 'PCA'

    def __str__(self) -> str:
        return 'PCA'

    def reduce(self, *args, **kwargs) -> None:
        self.instance = PCA(*args, **kwargs)
        return self.instance.fit_transform(self.dataset)


class TSNEStrategy(DimensionalityReductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return 't-SNE'

    def __str__(self) -> str:
        return 't-SNE'

    def reduce(self, *args, **kwargs) -> None:
        self.instance = TSNE(*args, **kwargs)
        return self.instance.fit_transform(self.dataset)


class UMAPStrategy(DimensionalityReductionStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return 'UMAP'

    def __str__(self) -> str:
        return 'UMAP'

    def reduce(self, *args, **kwargs) -> None:
        self.instance = UMAP(*args, **kwargs)
        return self.instance.fit_transform(self.dataset)
