import json
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


DATA_PATH = "heart_2020_cleaned.csv"
TARGET_COL = "HeartDisease"
MODEL_PATH = "mlp_model.pkl"
META_PATH = "mlp_meta.json"


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", MinMaxScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    return preprocessing


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df.drop(TARGET_COL, axis=1)
    # Encode target to numeric to avoid scoring issues inside MLP (np.isnan on strings)
    y = (df[TARGET_COL] == "Yes").astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    batch_size=1024,
                    learning_rate_init=0.001,
                    max_iter=20,
                    early_stopping=True,
                    n_iter_no_change=3,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values
    auc = roc_auc_score(y_true, proba)
    print(f"MLP ROC-AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)

    numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
    cat_choices: Dict[str, Any] = {
        col: sorted(df[col].dropna().unique().tolist()) for col in categorical_columns
    }
    meta = {
        "target": TARGET_COL,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_choices": cat_choices,
        "positive_class": "Yes",
        "target_mapping": {"No": 0, "Yes": 1},
        "model_path": MODEL_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved MLP model to {MODEL_PATH} and meta to {META_PATH}")


if __name__ == "__main__":
    main()


