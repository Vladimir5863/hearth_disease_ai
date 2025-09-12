from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def _build_auto_preprocessor(df: pd.DataFrame, target_column: str) -> ColumnTransformer:
    numeric_columns = df.drop(columns=[target_column]).select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.drop(columns=[target_column]).select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, list(numeric_columns)),
            ("categorical", categorical_transformer, list(categorical_columns)),
        ]
    )
    return preprocessing


def _to_binary_series(y: Union[pd.Series, np.ndarray]) -> pd.Series:
    if isinstance(y, pd.Series) and y.dtype == object:
        return (y == "Yes").astype(int)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return y.astype(int)


def _prob_or_score(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Positive class probability at index 1
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Map scores to [0,1] via logistic-like transform for AP/ROC (ranking unaffected)
        # Not strictly required for ROC-AUC, but keeps shape consistent
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    # Fallback to binary predictions if neither available
    preds = model.predict(X)
    return _to_binary_series(preds).values.astype(float)


@dataclass
class ModelResult:
    name: str
    best_params: Dict[str, object]
    metrics: Dict[str, float]
    report: str
    confusion: np.ndarray
    fitted: BaseEstimator


class BestModelSelector:
    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        param_grids: Dict[str, Dict[str, List[object]]],
        target_column: str = "HeartDisease",
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.param_grids = param_grids
        self.target_column = target_column
        self.random_state = random_state
        self.results: List[ModelResult] = []
        self.best_: Optional[ModelResult] = None

    def fit(self, df: pd.DataFrame, test_size: float = 0.3) -> "BestModelSelector":
        X = df.drop(self.target_column, axis=1)
        # Convert target to binary ints to avoid object dtype issues inside some estimators (e.g., MLP early_stopping)
        y = _to_binary_series(df[self.target_column])

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )

        preprocessing = _build_auto_preprocessor(df, self.target_column)

        best_overall: Optional[Tuple[float, ModelResult]] = None

        for name, estimator in self.estimators.items():
            est_to_use = clone(estimator)
            # Align XGBoost with notebooks: compute scale_pos_weight from y_train
            if name == "xgb":
                pos = int(pd.Series(y_train).sum())
                neg = int(len(y_train) - pos)
                scale = float(neg / pos) if pos > 0 else 1.0
                est_to_use.set_params(scale_pos_weight=scale)

            pipe = Pipeline(steps=[
                ("preprocessing", preprocessing),
                ("model", est_to_use),
            ])

            grid = self.param_grids.get(name, {})  # empty dict => single fit with given params
            gs = GridSearchCV(
                pipe,
                grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_pred = best_model.predict(X_test)
            # y_test is already binary (0/1)
            y_true_bin = pd.Series(y_test).astype(int)
            y_pred_bin = pd.Series(y_pred).astype(int)
            y_score = _prob_or_score(best_model, X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true_bin, y_score)),
                "pr_auc": float(average_precision_score(y_true_bin, y_score)),
                "mse": float(mean_squared_error(y_true_bin, y_pred_bin)),
                "r2": float(r2_score(y_true_bin, y_pred_bin)),
                "best_cv_roc_auc": float(gs.best_score_),
            }

            res = ModelResult(
                name=name,
                best_params=gs.best_params_,
                metrics=metrics,
                report=classification_report(y_true_bin, y_pred_bin, digits=4),
                confusion=confusion_matrix(y_true_bin, y_pred_bin),
                fitted=best_model,
            )
            self.results.append(res)

            score_for_selection = metrics["roc_auc"]
            if best_overall is None or score_for_selection > best_overall[0]:
                best_overall = (score_for_selection, res)

        if best_overall is not None:
            self.best_ = best_overall[1]
        return self

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"model": r.name}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)


def default_estimators_and_grids() -> Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, List[object]]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    estimators: Dict[str, BaseEstimator] = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "svm": LinearSVC(random_state=42, class_weight="balanced", max_iter=5000),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                              batch_size=1024, learning_rate_init=0.001, max_iter=50,
                              early_stopping=True, n_iter_no_change=3, random_state=42),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    grids: Dict[str, Dict[str, List[object]]] = {
        "logreg": {"model__C": [0.1, 1.0, 3.0], "model__solver": ["lbfgs"]},
        "svm": {"model__C": [0.1, 1.0, 3.0]},
        "mlp": {"model__hidden_layer_sizes": [(128, 64), (256, 128)], "model__learning_rate_init": [0.001, 0.0005]},
        "xgb": {
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.1, 0.05],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    }

    return estimators, grids


if __name__ == "__main__":
    # Example usage: run selection on the balanced CSV if present; otherwise fall back to original
    import os

    csv_path = "heart_2020_balanced.csv" if os.path.exists("heart_2020_balanced.csv") else "heart_2020_cleaned.csv"
    df = pd.read_csv(csv_path)

    ests, grids = default_estimators_and_grids()
    selector = BestModelSelector(ests, grids)
    selector.fit(df)
    print(selector.summary())
    if selector.best_ is not None:
        print("\nBest model:", selector.best_.name)
        print("Best params:", selector.best_.best_params)
        print("Metrics:", selector.best_.metrics)
        print("\nClassification report:\n", selector.best_.report)
        print("Confusion matrix:\n", selector.best_.confusion)


