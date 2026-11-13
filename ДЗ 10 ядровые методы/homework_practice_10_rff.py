import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func
        self.sigma_squared = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        X = np.asarray(X)
        n_obj, d = X.shape
        n_pairs = min(1_000_000, n_obj * (n_obj - 1) // 2)

        rng = np.random.default_rng()
        first = rng.integers(0, n_obj, size=n_pairs)
        second = rng.integers(0, n_obj - 1, size=n_pairs)
        second += second >= first # Этот хак позволяет избежать равных индексов в first и second и при этом в second все индексы равновероятно попадаются
        squared_dists = np.sum((X[first] - X[second]) ** 2, axis=1)
        sigma_squared = float(np.median(squared_dists))
        self.sigma_squared = sigma_squared

        self.w = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(sigma_squared),
            size=(d, self.n_features),
        )
        self.b = rng.uniform(-np.pi, np.pi, size=self.n_features)
        return self

    def transform(self, X, y=None):
        if self.w is None or self.b is None:
            raise RuntimeError("Сначала нужно вызвать fit")
        X = np.asarray(X)
        return self.func(X @ self.w + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        super().fit(X, y)
        rng = np.random.default_rng()
        n_dimensions = self.w.shape[0]
        blocks = []
        for start in range(0, self.n_features, n_dimensions):
            block_size = min(n_dimensions, self.n_features - start)
            q, _ = np.linalg.qr(
                rng.normal(size=(n_dimensions, n_dimensions))
            )
            norms = np.sqrt(
                rng.chisquare(n_dimensions, size=n_dimensions)
            )
            blocks.append((q * norms)[:, :block_size])
        self.w = np.concatenate(blocks, axis=1) / np.sqrt(self.sigma_squared)
        return self


class RFFPipeline(ClassifierMixin, BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.feature_creator_class = feature_creator_class
        self.func = func
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps: list[tuple] = []
        if self.use_PCA:
            pipeline_steps.append(("pca", PCA(n_components=self.new_dim)))

        pipeline_steps.append(("random_features", self.feature_creator))
        pipeline_steps.append(("classifier", self.classifier))
        
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError("Запустите fit перед этой функцией")
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Запустите fit перед этой функцией")
        return self.pipeline.predict(X)
