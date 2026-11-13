from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.best_iteration = None

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.grad_fn = lambda y, z: -y / (1 + np.exp(y * z))

    def partial_fit(self, X: np.ndarray, y: np.ndarray, train_predictions: np.ndarray, iter: int) -> None:
        s_i = -self.grad_fn(y, train_predictions)

        model = self.base_model_class(random_state=self.random_state, **self.base_model_params)
        model.fit(X, s_i)
        self.models[iter] = model

        new_predictions = model.predict(X)

        gamma = self._find_optimal_gamma(y, train_predictions, new_predictions)
        self.gammas[iter] = gamma

        train_predictions += self.learning_rate * gamma * new_predictions

        self.history["train_loss"].append(self.loss_fn(y, train_predictions))
        self.history['train_roc_auc'].append(roc_auc_score(y, self.sigmoid(train_predictions)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: tuple[np.ndarray] | None = None, use_best_model: bool = False) -> None:

        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        # Инициализируем валидационные предсказания, если eval_set предоставлен
        valid_predictions = None
        if eval_set is not None:
            X_valid, y_valid = eval_set
            valid_predictions = np.zeros(X_valid.shape[0])

        if self.eval_metric == 'valid_loss':
            best_score = float('inf')
        elif self.eval_metric == 'valid_roc_auc':
            best_score = float('-inf')
        else:
            best_score = None

        iter_without_improve = 0

        for iter in estimator_range:
            self.partial_fit(X_train, y_train, train_predictions, iter)

            if eval_set is not None:
                new_pred_valid = self.models[iter].predict(X_valid)

                valid_predictions += self.learning_rate * self.gammas[iter] * new_pred_valid

                valid_loss = self.loss_fn(y_valid, valid_predictions)
                self.history["valid_loss"].append(valid_loss)

                try:
                    valid_roc = roc_auc_score(y_valid, self.sigmoid(valid_predictions))
                except: # например, если всего один класс по y в выборке
                    valid_roc = float('nan')
                self.history['valid_roc_auc'].append(valid_roc)

                if self.eval_metric in ("valid_loss", "valid_roc_auc"): # Только эти обрабатываю
                    if self.eval_metric == "valid_loss":
                        improved = valid_loss < best_score
                        current_score = valid_loss
                    else:
                        improved = valid_roc > best_score
                        current_score = valid_roc

                    if improved:
                        best_score = current_score
                        iter_without_improve = 0
                        self.best_iteration = iter
                    else:
                        iter_without_improve += 1
                        if self.early_stopping_rounds and iter_without_improve >= self.early_stopping_rounds:
                            if self.verbose:
                                print(f'Early stopping on {iter} iteration')
                            break

        if use_best_model and self.best_iteration is not None:
            self.models = self.models[:self.best_iteration + 1]
            self.gammas = self.gammas[:self.best_iteration + 1]

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def plot_history(self, keys: str | list | tuple) -> None:
        """Plot training history for specified metrics.

        Parameters:
        -----------
        keys : str or Iterable[str]
            Key or keys from self.history to plot
        """
        import matplotlib.pyplot as plt

        if isinstance(keys, str):
            keys = [keys]

        plt.figure(figsize=(10, 6))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key, marker='o', markersize=3)

        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(X.shape[0])

        for model, gamma in zip(self.models, self.gammas):
            predictions += gamma * model.predict(X)

        predictions *= self.learning_rate

        proba_1 = self.sigmoid(predictions)

        return np.column_stack([1 - proba_1, proba_1])

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray, 
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

