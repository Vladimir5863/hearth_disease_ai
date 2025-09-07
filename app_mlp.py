import json
from typing import Dict, Any

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "mlp_model.pkl"
META_PATH = "mlp_meta.json"


def _infer_meta_from_model(model) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    pre = getattr(model, "named_steps", {}).get("preprocessing")
    if pre is not None and hasattr(pre, "transformers"):
        numeric_columns = []
        categorical_columns = []
        for name, transf, cols in pre.transformers:
            if name == "numerical":
                numeric_columns = list(cols)
            if name == "categorical":
                categorical_columns = list(cols)
        meta["numeric_columns"] = numeric_columns
        meta["categorical_columns"] = categorical_columns
    return meta


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta: Dict[str, Any] = {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    if "numeric_columns" not in meta or "categorical_columns" not in meta:
        inferred = _infer_meta_from_model(model)
        meta = {**inferred, **meta}

    if "categorical_choices" not in meta:
        try:
            df = pd.read_csv("heart_2020_cleaned.csv")
            choices: Dict[str, Any] = {}
            for col in meta.get("categorical_columns", []):
                if col in df.columns:
                    choices[col] = sorted(df[col].dropna().unique().tolist())
            meta["categorical_choices"] = choices
        except Exception:
            meta.setdefault("categorical_choices", {})

    return model, meta


def build_input_df(meta: Dict[str, Any]) -> pd.DataFrame:
    st.sidebar.header("Input features")

    num_vals: Dict[str, float] = {}
    for col in meta.get("numeric_columns", []):
        val = st.sidebar.number_input(col, value=0.0)
        num_vals[col] = float(val)

    cat_vals: Dict[str, Any] = {}
    for col in meta.get("categorical_columns", []):
        choices = meta.get("categorical_choices", {}).get(col, [])
        val = st.sidebar.selectbox(col, choices, index=0 if choices else None)
        cat_vals[col] = val

    row = {**num_vals, **cat_vals}
    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title="Heart Disease Risk - MLP", layout="centered")
    st.title("Heart Disease Risk - MLP")

    try:
        model, meta = load_model()
    except Exception as e:
        st.error(f"Failed to load model/meta: {e}")
        st.stop()

    st.markdown("Model loaded. Enter features in the sidebar and click Predict.")

    X = build_input_df(meta)

    if st.button("Predict"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        st.subheader("Prediction")
        st.write(f"Probability of HeartDisease=Yes: {float(proba[0]):.4f}")
        st.write(f"Predicted label: {'Yes' if pred[0] == 1 else 'No'}")


if __name__ == "__main__":
    main()


