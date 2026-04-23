"""From-scratch logistic regression utilities used in the notebooks.

The original notebooks repeated the same cells for loading data,
standardizing features, fitting logistic regression with gradient descent,
predicting classes, and plotting simple diagnostics. This module keeps that
logic in reusable functions and classes without relying on scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, overload

import numpy as np
import pandas as pd


ArrayLike = Sequence[float] | np.ndarray | pd.Series | pd.DataFrame


def _as_2d_float_array(X: ArrayLike) -> np.ndarray:
    """Return ``X`` as a two-dimensional float NumPy array."""
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    return array


def _as_1d_array(y: ArrayLike) -> np.ndarray:
    """Return ``y`` as a one-dimensional NumPy array."""
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    return array


@overload
def sigmoid(z: float) -> float:
    ...


@overload
def sigmoid(z: ArrayLike) -> np.ndarray:
    ...


def sigmoid(z: ArrayLike | float) -> np.ndarray | float:
    """Compute the logistic sigmoid function in a numerically stable way.

    Parameters
    ----------
    z:
        Scalar or array-like input.

    Returns
    -------
    numpy.ndarray | float
        Values transformed to the interval ``(0, 1)``.
    """
    z_array = np.asarray(z, dtype=float)
    result = np.empty_like(z_array, dtype=float)

    positive_mask = z_array >= 0
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z_array[positive_mask]))

    negative_mask = ~positive_mask
    exp_z = np.exp(z_array[negative_mask])
    result[negative_mask] = exp_z / (1.0 + exp_z)

    if np.isscalar(z):
        return float(result)
    return result


def binary_cross_entropy(
    X: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike,
    bias: float,
    eps: float = 1e-15,
) -> float:
    """Compute average binary cross-entropy for logistic regression.

    This is the vectorized equivalent of the ``cost_function`` repeated in
    the notebooks.
    """
    X_array = _as_2d_float_array(X)
    y_array = np.asarray(y, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if y_array.shape != (X_array.shape[0],):
        raise ValueError("y length must match the number of rows in X.")
    if weights_array.shape != (X_array.shape[1],):
        raise ValueError("weights length must match the number of features.")

    probabilities = np.asarray(sigmoid(X_array @ weights_array + bias), dtype=float)
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    loss = -y_array * np.log(probabilities) - (1.0 - y_array) * np.log(
        1.0 - probabilities
    )
    return float(np.mean(loss))


def logistic_gradient(
    X: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike,
    bias: float,
) -> tuple[np.ndarray, float]:
    """Compute gradients of binary cross-entropy over weights and bias."""
    X_array = _as_2d_float_array(X)
    y_array = np.asarray(y, dtype=float)
    weights_array = np.asarray(weights, dtype=float)

    if y_array.shape != (X_array.shape[0],):
        raise ValueError("y length must match the number of rows in X.")
    if weights_array.shape != (X_array.shape[1],):
        raise ValueError("weights length must match the number of features.")

    probabilities = np.asarray(sigmoid(X_array @ weights_array + bias), dtype=float)
    errors = probabilities - y_array
    grad_weights = (X_array.T @ errors) / X_array.shape[0]
    grad_bias = float(np.mean(errors))
    return grad_weights, grad_bias


def standardize(X: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize ``X`` and return ``(X_scaled, mean, scale)``.

    This mirrors the notebook cells that computed ``mu``, ``sigma`` and
    ``X_train = (X - mu) / sigma``.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if scaler.mean_ is None or scaler.scale_ is None:
        raise RuntimeError("Scaler was not fitted.")
    return X_scaled, scaler.mean_, scaler.scale_


def cost_function(X: ArrayLike, y: ArrayLike, w: ArrayLike, b: float) -> float:
    """Notebook-compatible alias for binary logistic regression cost."""
    return binary_cross_entropy(X, y, w, b)


def gradient_function(
    X: ArrayLike,
    y: ArrayLike,
    w: ArrayLike,
    b: float,
) -> tuple[float, np.ndarray]:
    """Notebook-compatible gradient function.

    The return order is ``(grad_b, grad_w)`` to match the original notebooks.
    """
    grad_w, grad_b = logistic_gradient(X, y, w, b)
    return grad_b, grad_w


def gradient_descent(
    X: ArrayLike,
    y: ArrayLike,
    alpha: float,
    iterations: int,
    *,
    print_every: int | None = 1_000,
) -> tuple[np.ndarray, float]:
    """Run binary logistic gradient descent and return ``(weights, bias)``.

    This preserves the simple function interface used in the notebooks.
    """
    X_array = _as_2d_float_array(X)
    y_array = np.asarray(y, dtype=float)
    w = np.zeros(X_array.shape[1], dtype=float)
    b = 0.0

    for i in range(iterations):
        grad_b, grad_w = gradient_function(X_array, y_array, w, b)
        w -= alpha * grad_w
        b -= alpha * grad_b

        if print_every is not None and i % print_every == 0:
            print(f"Iteration {i}: Cost {cost_function(X_array, y_array, w, b)}")

    return w, b


def predict(
    X: ArrayLike,
    w: ArrayLike,
    b: float,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Notebook-compatible binary prediction function returning ``0`` or ``1``."""
    X_array = _as_2d_float_array(X)
    weights = np.asarray(w, dtype=float)
    probabilities = np.asarray(sigmoid(X_array @ weights + b), dtype=float)
    return (probabilities >= threshold).astype(int)


@dataclass
class StandardScaler:
    """Standardize features with the z-score used in the notebooks.

    Constant columns are left unchanged by replacing their standard deviation
    with ``1``. This avoids division-by-zero while preserving column shape.
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: ArrayLike) -> "StandardScaler":
        """Estimate mean and standard deviation for each feature column."""
        X_array = _as_2d_float_array(X)
        self.mean_ = np.mean(X_array, axis=0)
        scale = np.std(X_array, axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply z-score standardization using the fitted statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        X_array = _as_2d_float_array(X)
        return (X_array - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit the scaler and return standardized features."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Undo standardization and return values on the original scale."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before inverse_transform.")
        X_array = _as_2d_float_array(X)
        return X_array * self.scale_ + self.mean_


class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    learning_rate:
        Gradient descent step size.
    iterations:
        Maximum number of optimization iterations.
    tolerance:
        Optional early stopping threshold for consecutive cost changes. Use
        ``None`` to always run all iterations.
    print_every:
        If provided, print the cost every ``print_every`` iterations.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 10_000,
        tolerance: float | None = None,
        print_every: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.print_every = print_every

        self.weights_: np.ndarray | None = None
        self.bias_: float | None = None
        self.classes_: np.ndarray | None = None
        self.cost_history_: list[float] = []
        self.n_iterations_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegressionGD":
        """Fit model parameters on binary labels.

        ``y`` may contain any two distinct labels. Internally they are mapped
        to ``0`` and ``1``; predictions are mapped back to the original labels.
        """
        X_array = _as_2d_float_array(X)
        y_array = _as_1d_array(y)

        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("y length must match the number of rows in X.")

        classes = np.unique(y_array)
        if classes.shape[0] != 2:
            raise ValueError(
                "LogisticRegressionGD supports exactly two classes. "
                "Use OneVsRestLogisticRegressionGD for multiclass targets."
            )

        self.classes_ = classes
        y_binary = np.equal(y_array, classes[1]).astype(float)
        weights = np.zeros(X_array.shape[1], dtype=float)
        bias = 0.0
        self.cost_history_ = []

        for i in range(self.iterations):
            grad_weights, grad_bias = logistic_gradient(X_array, y_binary, weights, bias)
            weights -= self.learning_rate * grad_weights
            bias -= self.learning_rate * grad_bias

            cost = binary_cross_entropy(X_array, y_binary, weights, bias)
            self.cost_history_.append(cost)

            if self.print_every is not None and i % self.print_every == 0:
                print(f"Iteration {i}: Cost {cost}")

            if (
                self.tolerance is not None
                and i > 0
                and abs(self.cost_history_[-2] - self.cost_history_[-1])
                < self.tolerance
            ):
                self.n_iterations_ = i + 1
                break
        else:
            self.n_iterations_ = self.iterations

        self.weights_ = weights
        self.bias_ = bias
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return raw linear scores before the sigmoid transformation."""
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X_array = _as_2d_float_array(X)
        return X_array @ self.weights_ + self.bias_

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probability estimates for the positive class."""
        probabilities = sigmoid(self.decision_function(X))
        return np.asarray(probabilities, dtype=float)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels using a probability threshold."""
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        positive = self.predict_proba(X) >= threshold
        return np.where(positive, self.classes_[1], self.classes_[0])

    def score(self, X: ArrayLike, y: ArrayLike, threshold: float = 0.5) -> float:
        """Return classification accuracy."""
        y_array = _as_1d_array(y)
        predictions = self.predict(X, threshold=threshold)
        if y_array.shape != predictions.shape:
            raise ValueError("y length must match the number of predictions.")
        return float(np.mean(predictions == y_array))


class OneVsRestLogisticRegressionGD:
    """Multiclass logistic regression built from binary one-vs-rest models."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 10_000,
        tolerance: float | None = None,
        print_every: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.print_every = print_every

        self.classes_: np.ndarray | None = None
        self.models_: dict[Any, LogisticRegressionGD] = {}

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OneVsRestLogisticRegressionGD":
        """Fit one binary classifier for each target class."""
        X_array = _as_2d_float_array(X)
        y_array = _as_1d_array(y)

        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("y length must match the number of rows in X.")

        self.classes_ = np.unique(y_array)
        self.models_ = {}

        for target_class in self.classes_:
            binary_y = np.equal(y_array, target_class).astype(int)
            model = LogisticRegressionGD(
                learning_rate=self.learning_rate,
                iterations=self.iterations,
                tolerance=self.tolerance,
                print_every=self.print_every,
            )
            model.fit(X_array, binary_y)
            self.models_[target_class] = model

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return one-vs-rest probabilities with shape ``(n_samples, n_classes)``."""
        if self.classes_ is None or not self.models_:
            raise ValueError("Model must be fitted before prediction.")
        columns = [self.models_[target_class].predict_proba(X) for target_class in self.classes_]
        return np.column_stack(columns)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the highest one-vs-rest probability."""
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        y_array = _as_1d_array(y)
        predictions = self.predict(X)
        if y_array.shape != predictions.shape:
            raise ValueError("y length must match the number of predictions.")
        return float(np.mean(predictions == y_array))


@dataclass
class LogisticModelTreeNode:
    """Single node placeholder for a future Logistic Model Tree.

    A leaf node will store a fitted logistic regression model. An internal
    node will store a split rule and references to its child nodes.
    """

    depth: int
    is_leaf: bool = True
    model: LogisticRegressionGD | OneVsRestLogisticRegressionGD | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "LogisticModelTreeNode | None" = None
    right: "LogisticModelTreeNode | None" = None
    n_samples: int = 0
    impurity: float | None = None


class LogisticModelTree:
    """Placeholder for a Logistic Model Tree classifier.

    The intended algorithm is a decision tree where every node trains a
    logistic regression model using the current subset of rows. Internal nodes
    choose a split and pass samples to children; leaf nodes keep the final
    logistic model used for predictions.

    This class is intentionally not implemented yet. It documents the API and
    the concrete implementation steps needed later.
    """

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        learning_rate: float = 0.01,
        iterations: int = 10_000,
        tolerance: float | None = None,
        print_every: int | None = None,
    ) -> None:
        """Create a Logistic Model Tree configuration.

        Parameters
        ----------
        max_depth:
            Maximum tree depth. Depth ``0`` is the root.
        min_samples_split:
            Minimum number of rows required to consider splitting a node.
        min_samples_leaf:
            Minimum number of rows allowed in each child after a split.
        learning_rate, iterations, tolerance, print_every:
            Parameters that should be passed into the logistic regression
            model trained at every node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.print_every = print_every

        self.root_: LogisticModelTreeNode | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticModelTree":
        """Fit the full Logistic Model Tree.

        TODO:
        1. Convert ``X`` to a 2D float array using ``_as_2d_float_array``.
        2. Convert ``y`` to a 1D array using ``_as_1d_array``.
        3. Validate that ``X`` and ``y`` contain the same number of rows.
        4. Store ``self.classes_ = np.unique(y)`` and ``self.n_features_``.
        5. Build the tree recursively by calling ``self._build_node`` on the
           full dataset with ``depth=0``.
        6. Store the returned node as ``self.root_``.
        7. Return ``self`` so the class behaves like the existing regression
           classes.
        """
        raise NotImplementedError("LogisticModelTree.fit is a scaffold only.")

    def _build_node(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        depth: int,
    ) -> LogisticModelTreeNode:
        """Build one tree node recursively.

        TODO:
        1. Create a ``LogisticModelTreeNode`` with the current ``depth`` and
           ``n_samples=len(y)``.
        2. Train a logistic regression model on this node's subset by calling
           ``self._fit_node_model(X, y)`` and store it in ``node.model``.
        3. Compute the current node quality/impurity. Possible choices:
           binary cross-entropy from the fitted model, Gini impurity, entropy,
           or classification error.
        4. Check stopping rules:
           - depth reached ``self.max_depth``;
           - fewer than ``self.min_samples_split`` rows;
           - all labels are the same;
           - no valid split improves the score.
        5. If a stopping rule is met, mark the node as a leaf and return it.
        6. Otherwise call ``self._find_best_split(X, y)``.
        7. Save the chosen ``feature_index`` and ``threshold`` in the node.
        8. Split rows into left/right masks using ``X[:, feature_index] <=
           threshold``.
        9. Recursively build ``node.left`` and ``node.right`` with
           ``depth + 1``.
        10. Mark the node as non-leaf and return it.
        """
        raise NotImplementedError("LogisticModelTree._build_node is a scaffold only.")

    def _fit_node_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> LogisticRegressionGD | OneVsRestLogisticRegressionGD:
        """Train the logistic regression model stored inside one node.

        TODO:
        1. Check how many unique classes exist in ``y``.
        2. If there are exactly two classes, create ``LogisticRegressionGD``
           with this tree's learning parameters and fit it on ``X, y``.
        3. If there are more than two classes, create
           ``OneVsRestLogisticRegressionGD`` with the same learning parameters
           and fit it on ``X, y``.
        4. Return the fitted model.
        5. Decide what to do if a node receives only one class. Practical
           options:
           - store a simple constant predictor class;
           - skip fitting and let the node become a leaf;
           - extend ``LogisticRegressionGD`` to support constant leaves.
        """
        raise NotImplementedError(
            "LogisticModelTree._fit_node_model is a scaffold only."
        )

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int, float, float] | None:
        """Find the best feature and threshold for splitting one node.

        TODO:
        1. Iterate over each feature column.
        2. Generate candidate thresholds. Simple first version:
           use sorted unique values and test midpoints between neighbors.
           Later optimization: use quantiles to reduce cost.
        3. For each candidate threshold, split rows into left and right.
        4. Reject splits where either child has fewer than
           ``self.min_samples_leaf`` rows.
        5. Score the split. Reasonable first scoring options:
           - weighted child cross-entropy after fitting child logistic models;
           - information gain using entropy/Gini on labels;
           - reduction in node logistic loss.
        6. Keep the split with the best improvement.
        7. Return ``(feature_index, threshold, score)``.
        8. Return ``None`` if no valid split exists.
        """
        raise NotImplementedError(
            "LogisticModelTree._find_best_split is a scaffold only."
        )

    def _should_stop(
        self,
        y: np.ndarray,
        *,
        depth: int,
    ) -> bool:
        """Decide whether a node should stop growing before split search.

        TODO:
        1. Return ``True`` when ``depth >= self.max_depth``.
        2. Return ``True`` when ``len(y) < self.min_samples_split``.
        3. Return ``True`` when ``np.unique(y)`` contains only one class.
        4. Optionally add more criteria later:
           - logistic loss is already low enough;
           - class distribution is too pure;
           - further split improvement is below a tolerance.
        5. Return ``False`` otherwise.
        """
        raise NotImplementedError(
            "LogisticModelTree._should_stop is a scaffold only."
        )

    def _route_sample(
        self,
        x: np.ndarray,
        node: LogisticModelTreeNode,
    ) -> LogisticModelTreeNode:
        """Route one sample from a node down to the leaf used for prediction.

        TODO:
        1. Start from the provided ``node``.
        2. While the current node is not a leaf:
           - read ``current.feature_index`` and ``current.threshold``;
           - if ``x[feature_index] <= threshold``, go to ``current.left``;
           - otherwise go to ``current.right``.
        3. Validate that required child nodes exist before following them.
        4. Return the final leaf node.
        """
        raise NotImplementedError(
            "LogisticModelTree._route_sample is a scaffold only."
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities with the leaf model for each sample.

        TODO:
        1. Validate that ``self.root_`` is not ``None``.
        2. Convert ``X`` to a 2D float array.
        3. For each row:
           - route it to a leaf using ``self._route_sample``;
           - call that leaf's logistic model ``predict_proba`` on the single
             row;
           - store the returned probabilities.
        4. Combine all row probabilities into one 2D array.
        5. For binary ``LogisticRegressionGD``, decide whether the API should
           return only positive-class probability or two columns
           ``[P(class0), P(class1)]``. A two-column output is usually easier
           for tree-level consistency.
        """
        raise NotImplementedError(
            "LogisticModelTree.predict_proba is a scaffold only."
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels by routing each row to a leaf logistic model.

        TODO:
        1. Call ``self.predict_proba(X)``.
        2. For binary classification, map the highest probability back to
           ``self.classes_``.
        3. For multiclass classification, use ``np.argmax`` over probability
           columns and map indexes back to ``self.classes_``.
        4. Return a 1D array of predicted labels.
        """
        raise NotImplementedError("LogisticModelTree.predict is a scaffold only.")

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy for the Logistic Model Tree.

        TODO:
        1. Convert ``y`` to a 1D array.
        2. Call ``self.predict(X)``.
        3. Validate that prediction length matches ``y`` length.
        4. Return ``np.mean(predictions == y)`` as a float.
        """
        raise NotImplementedError("LogisticModelTree.score is a scaffold only.")


@dataclass
class ManualPCA:
    """Principal component analysis implemented with NumPy eigenvectors."""

    n_components: int = 2
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: ArrayLike) -> "ManualPCA":
        """Estimate principal components from centered input features."""
        X_array = _as_2d_float_array(X)
        if not 1 <= self.n_components <= X_array.shape[1]:
            raise ValueError("n_components must be between 1 and the number of features.")

        self.mean_ = np.mean(X_array, axis=0)
        centered = X_array - self.mean_
        covariance = np.atleast_2d(np.cov(centered, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]

        sorted_values = eigenvalues[order]
        sorted_vectors = eigenvectors[:, order]

        components = sorted_vectors[:, : self.n_components]
        explained_variance = sorted_values[: self.n_components]

        self.components_ = components
        self.explained_variance_ = explained_variance
        total_variance = float(np.sum(sorted_values))
        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = explained_variance / total_variance
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Project input features onto the fitted principal components."""
        if self.mean_ is None or self.components_ is None:
            raise ValueError("ManualPCA must be fitted before transform.")
        X_array = _as_2d_float_array(X)
        return (X_array - self.mean_) @ self.components_

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit PCA and return projected features."""
        return self.fit(X).transform(X)


def manual_pca_2d(X: ArrayLike) -> np.ndarray:
    """Project standardized data to two principal components with NumPy only."""
    return ManualPCA(n_components=2).fit_transform(X)


@dataclass
class DatasetBundle:
    """Container returned by CSV preparation helpers."""

    X: np.ndarray
    y: np.ndarray
    dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    scaler: StandardScaler | None = None


def load_classification_csv(
    path: str | Path,
    target_column: str,
    *,
    drop_columns: Sequence[str] | None = None,
    target_mapping: Mapping[Any, Any] | None = None,
    feature_mappings: Mapping[str, Mapping[Any, Any]] | None = None,
    na_values: Any | None = None,
    fillna_mode: bool = False,
    dropna: bool = True,
    standardize: bool = True,
) -> DatasetBundle:
    """Load a CSV dataset and prepare ``X``/``y`` for the regression classes.

    The options cover the unique preprocessing cells from the notebooks:
    dropping unused columns, mapping labels such as ``N/P`` or ``y/n``, mode
    imputation, optional ``?`` NA handling, and z-score standardization.
    """
    read_kwargs: dict[str, Any] = {}
    if na_values is not None:
        read_kwargs["na_values"] = na_values

    dataframe = pd.read_csv(path, **read_kwargs)
    dataframe = prepare_dataframe(
        dataframe,
        target_column=target_column,
        drop_columns=drop_columns,
        target_mapping=target_mapping,
        feature_mappings=feature_mappings,
        fillna_mode=fillna_mode,
        dropna=dropna,
    )

    feature_columns = [column for column in dataframe.columns if column != target_column]
    X = dataframe[feature_columns].to_numpy(dtype=float)
    y = dataframe[target_column].to_numpy()

    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return DatasetBundle(
        X=X,
        y=y,
        dataframe=dataframe,
        feature_columns=feature_columns,
        target_column=target_column,
        scaler=scaler,
    )


def prepare_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    drop_columns: Sequence[str] | None = None,
    target_mapping: Mapping[Any, Any] | None = None,
    feature_mappings: Mapping[str, Mapping[Any, Any]] | None = None,
    fillna_mode: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """Apply notebook-style cleaning and categorical value mappings."""
    prepared = dataframe.copy()

    if drop_columns:
        prepared = prepared.drop(columns=list(drop_columns))

    if fillna_mode:
        prepared = prepared.fillna(prepared.mode().iloc[0])

    if target_mapping is not None:
        prepared[target_column] = prepared[target_column].map(target_mapping)

    if feature_mappings:
        for column, mapping in feature_mappings.items():
            prepared[column] = prepared[column].map(mapping)

    if dropna:
        prepared = prepared.dropna()

    return prepared


def prepare_vote_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare the vote dataset from the outlier notebook.

    Missing values are filled with the mode, vote columns are mapped from
    ``y/n`` to ``1/0``, and the class is mapped to ``democrat=0`` and
    ``republican=1``.
    """
    prepared = dataframe.copy()
    prepared = prepared.fillna(prepared.mode().iloc[0])
    vote_mapping = {"y": 1, "n": 0}

    for column in prepared.columns.drop("Class"):
        prepared[column] = prepared[column].map(vote_mapping)

    prepared["Class"] = prepared["Class"].map({"democrat": 0, "republican": 1})
    return prepared.dropna()


def train_binary_logistic_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    learning_rate: float = 0.01,
    iterations: int = 10_000,
    tolerance: float | None = None,
    print_every: int | None = None,
) -> LogisticRegressionGD:
    """Convenience wrapper matching the training cells in the notebooks."""
    model = LogisticRegressionGD(
        learning_rate=learning_rate,
        iterations=iterations,
        tolerance=tolerance,
        print_every=print_every,
    )
    return model.fit(X, y)


def plot_binary_scatter(
    X: ArrayLike,
    y: ArrayLike,
    *,
    feature_names: Sequence[str] = ("Feature1", "Feature2"),
    class_labels: Sequence[str] = ("Class 0", "Class 1"),
    ax: Any | None = None,
) -> Any:
    """Plot the first two standardized features colored by binary class."""
    import matplotlib.pyplot as plt

    X_array = _as_2d_float_array(X)
    y_array = _as_1d_array(y)
    if X_array.shape[1] < 2:
        raise ValueError("At least two features are required for a scatter plot.")

    classes = np.unique(y_array)
    if classes.shape[0] != 2:
        raise ValueError("plot_binary_scatter expects exactly two classes.")

    if ax is None:
        _, ax = plt.subplots()

    colors = ("tab:blue", "tab:orange")
    for label, color, class_name in zip(classes, colors, class_labels):
        mask = y_array == label
        ax.scatter(X_array[mask, 0], X_array[mask, 1], color=color, label=class_name, s=20)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    return ax


def plot_decision_boundary_2d(
    X: ArrayLike,
    y: ArrayLike,
    model: LogisticRegressionGD,
    *,
    feature_names: Sequence[str] = ("Feature1", "Feature2"),
    ax: Any | None = None,
) -> Any:
    """Plot a fitted two-feature logistic regression decision boundary."""
    import matplotlib.pyplot as plt

    if model.weights_ is None or model.bias_ is None:
        raise ValueError("Model must be fitted before plotting.")

    X_array = _as_2d_float_array(X)
    if X_array.shape[1] != 2:
        raise ValueError("Decision boundary plot requires exactly two features.")
    if abs(model.weights_[1]) < 1e-12:
        raise ValueError("Cannot plot boundary because the second weight is near zero.")

    if ax is None:
        _, ax = plt.subplots()

    xmin, xmax = X_array[:, 0].min() - 0.5, X_array[:, 0].max() + 0.5
    ymin, ymax = X_array[:, 1].min() - 0.5, X_array[:, 1].max() + 0.5
    xd = np.array([xmin, xmax])
    slope = -model.weights_[0] / model.weights_[1]
    intercept = -model.bias_ / model.weights_[1]
    yd = slope * xd + intercept

    ax.plot(xd, yd, "k", ls="--", label="Decision Boundary")
    ax.fill_between(xd, yd, ymin, color="tab:blue", alpha=0.2)
    ax.fill_between(xd, yd, ymax, color="tab:orange", alpha=0.2)
    plot_binary_scatter(X_array, y, feature_names=feature_names, ax=ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax


def plot_framingham_views(
    dataframe: pd.DataFrame,
    *,
    target_column: str = "TenYearCHD",
    ax: Sequence[Any] | None = None,
) -> Sequence[Any]:
    """Recreate the three Framingham exploratory plots from the notebook."""
    import matplotlib.pyplot as plt

    required = {target_column, "age", "sysBP", "totChol", "BMI"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required Framingham columns: {missing}")

    df = dataframe.dropna()
    y = df[target_column].to_numpy()
    X = df.drop(target_column, axis=1).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = ManualPCA(n_components=2).fit_transform(X_scaled)

    if ax is None:
        _, axes_array = plt.subplots(1, 3, figsize=(18, 5))
        axes = list(np.ravel(axes_array))
    else:
        axes = list(ax)
        if len(axes) != 3:
            raise ValueError("ax must contain exactly three axes.")

    negative = df[target_column] == 0
    positive = df[target_column] == 1

    axes[0].scatter(df.loc[negative, "age"], df.loc[negative, "sysBP"], alpha=0.5, color="#1f77b4", label="Class 0")
    axes[0].scatter(df.loc[positive, "age"], df.loc[positive, "sysBP"], alpha=0.5, color="#ff7f0e", label="Class 1")
    axes[0].set_title("Age vs Systolic Blood Pressure")
    axes[0].set_xlabel("age")
    axes[0].set_ylabel("sysBP")
    axes[0].legend()

    axes[1].scatter(df.loc[negative, "totChol"], df.loc[negative, "BMI"], alpha=0.5, color="#1f77b4", label="Class 0")
    axes[1].scatter(df.loc[positive, "totChol"], df.loc[positive, "BMI"], alpha=0.5, color="#ff7f0e", label="Class 1")
    axes[1].set_title("Cholesterol vs BMI")
    axes[1].set_xlabel("totChol")
    axes[1].set_ylabel("BMI")
    axes[1].legend()

    axes[2].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.5, color="#1f77b4", label="Class 0")
    axes[2].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.5, color="#ff7f0e", label="Class 1")
    axes[2].set_title("Manual PCA")
    axes[2].set_xlabel("Principal Component 1")
    axes[2].set_ylabel("Principal Component 2")
    axes[2].legend()

    return axes
