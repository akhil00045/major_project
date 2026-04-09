import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CKD Interactive App", layout="wide")

DATA_PATH = "kidney_disease.csv"
TARGET_COL = "class"

RENAME_MAP = {
    "bp": "blood_pressure",
    "sg": "specific_gravity",
    "al": "albumin",
    "su": "sugar",
    "rbc": "red_blood_cells",
    "pc": "pus_cell",
    "pcc": "pus_cell_clumps",
    "ba": "bacteria",
    "bgr": "blood_glucose_random",
    "bu": "blood_urea",
    "sc": "serum_creatinine",
    "sod": "sodium",
    "pot": "potassium",
    "hemo": "haemoglobin",
    "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "htn": "hypertension",
    "dm": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "appet": "appetite",
    "pe": "peda_edema",
    "ane": "aanemia",
    "classification": "class",
}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _random_value_imputation(df: pd.DataFrame, feature: str, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    null_count = out[feature].isna().sum()
    if null_count == 0:
        return out
    random_sample = out[feature].dropna().sample(null_count, random_state=seed, replace=True)
    random_sample.index = out[out[feature].isnull()].index
    out.loc[out[feature].isnull(), feature] = random_sample
    return out


@st.cache_data
def preprocess_data(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, list], Dict[str, LabelEncoder]]:
    df = raw_df.copy()
    missing_before = {
        "total": df.isna().sum().sort_values(ascending=False),
    }

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.rename(columns=RENAME_MAP)

    for col in ["packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "diabetes_mellitus" in df.columns:
        df["diabetes_mellitus"] = df["diabetes_mellitus"].replace(
            to_replace={"\tno": "no", "\tyes": "yes", " yes": "yes"}
        )
    if "coronary_artery_disease" in df.columns:
        df["coronary_artery_disease"] = df["coronary_artery_disease"].replace(to_replace="\tno", value="no")
    if "class" in df.columns:
        df["class"] = df["class"].replace(to_replace={"ckd\t": "ckd", "notckd": "not ckd"})
        df["class"] = df["class"].map({"ckd": 0, "not ckd": 1})
        df["class"] = pd.to_numeric(df["class"], errors="coerce")

    cat_cols = [col for col in df.columns if df[col].dtype == "object"]
    num_cols = [col for col in df.columns if df[col].dtype != "object"]

    for col in num_cols:
        df = _random_value_imputation(df, col)

    for col in ["red_blood_cells", "pus_cell"]:
        if col in df.columns:
            df = _random_value_imputation(df, col)

    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    label_encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        label_encoders[col] = encoder

    # Safety net: model inputs must be fully numeric.
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col].astype(str))[0]

    missing_after = {
        "total": df.isna().sum().sort_values(ascending=False),
    }

    metadata = {
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
    }
    return df, missing_before, metadata | {"missing_after_total": missing_after["total"]}, label_encoders


def get_model_registry(random_state: int = 0):
    return {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Stochastic Gradient Boosting": GradientBoostingClassifier(
            max_depth=4, subsample=0.90, max_features=0.75, n_estimators=200, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic", learning_rate=0.5, max_depth=5, n_estimators=150, random_state=random_state
        ),
        "CatBoost": CatBoostClassifier(iterations=50, verbose=0, random_state=random_state),
        "Extra Trees": ExtraTreesClassifier(random_state=random_state),
        "LightGBM": LGBMClassifier(learning_rate=0.1, random_state=random_state, verbose=-1),
    }


def train_models(df: pd.DataFrame, test_size: float, random_state: int):
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    x = df[feature_cols].copy()
    y = df[TARGET_COL]

    # Extra guard in case session/cache contains stale preprocessed data.
    # Any remaining text columns are encoded deterministically before model fitting.
    for col in x.select_dtypes(include=["object", "string", "category"]).columns:
        local_encoder = LabelEncoder()
        x[col] = local_encoder.fit_transform(x[col].astype(str))

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = get_model_registry(random_state=random_state)
    summary_rows = []
    details = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, output_dict=False)

        summary_rows.append(
            {
                "Model": name,
                "Train Accuracy": round(train_acc, 4),
                "Test Accuracy": round(test_acc, 4),
            }
        )
        details[name] = {
            "confusion_matrix": cm,
            "classification_report": report,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
    return summary_df, details, models


st.title("Chronic Kidney Disease Prediction App")
st.write("Notebook-to-Streamlit conversion with interactive EDA, preprocessing, and baseline modeling.")

raw_df = load_data(DATA_PATH)

tab_eda, tab_pre, tab_models, tab_predict = st.tabs(["EDA", "Preprocessing", "Models", "🔬 Patient Predictor"])

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.write(f"Raw shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(), width="stretch")

    eda_df = raw_df.rename(columns=RENAME_MAP).copy()
    if "id" in eda_df.columns:
        eda_df = eda_df.drop(columns=["id"])

    numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in eda_df.columns if c not in numeric_cols]

    c1, c2 = st.columns(2)
    with c1:
        num_feature = st.selectbox("Numeric feature distribution", options=numeric_cols, index=0)
        fig_hist = px.histogram(eda_df, x=num_feature, nbins=30, title=f"Distribution: {num_feature}")
        st.plotly_chart(fig_hist, width="stretch")
    with c2:
        cat_feature = st.selectbox("Categorical feature count", options=categorical_cols, index=0)
        fig_count = px.histogram(eda_df, x=cat_feature, title=f"Category counts: {cat_feature}")
        st.plotly_chart(fig_count, width="stretch")

    if "classification" in raw_df.columns:
        scatter_x = st.selectbox("Scatter X feature", options=numeric_cols, index=0)
        scatter_y = st.selectbox("Scatter Y feature", options=numeric_cols, index=min(1, len(numeric_cols) - 1))
        fig_scatter = px.scatter(
            eda_df,
            x=scatter_x,
            y=scatter_y,
            color=raw_df["classification"].astype(str),
            title=f"{scatter_x} vs {scatter_y}",
        )
        st.plotly_chart(fig_scatter, width="stretch")

    corr_df = eda_df.copy()
    for col in corr_df.columns:
        if corr_df[col].dtype == "object":
            corr_df[col] = corr_df[col].astype("category").cat.codes
    fig_corr = px.imshow(corr_df.corr(numeric_only=True), title="Correlation Heatmap")
    st.plotly_chart(fig_corr, width="stretch")

with tab_pre:
    st.subheader("Data Preprocessing")
    st.write("Click the button to run cleaning + encoding pipeline and save output in session state.")

    if st.button("Run preprocessing", type="primary"):
        clean_df, missing_before, metadata, label_encoders = preprocess_data(raw_df)
        st.session_state["clean_df"] = clean_df
        st.session_state["meta"] = metadata
        st.session_state["label_encoders"] = label_encoders
        st.session_state["missing_before"] = missing_before["total"]
        st.success("Preprocessing complete. `clean_df` stored in session state.")

    if "clean_df" in st.session_state:
        clean_df = st.session_state["clean_df"]
        st.write(f"Cleaned shape: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
        st.dataframe(clean_df.head(), width="stretch")

        st.write("Missing values (before preprocessing)")
        st.dataframe(st.session_state.get("missing_before").to_frame("missing_count"), width="stretch")
        st.write("Missing values (after preprocessing)")
        st.dataframe(
            clean_df.isna().sum().sort_values(ascending=False).to_frame("missing_count"), width="stretch"
        )

        st.write("Categorical columns (from original dtype split)")
        st.write(st.session_state["meta"]["categorical_columns"])
        st.write("Numerical columns")
        st.write(st.session_state["meta"]["numerical_columns"])
    else:
        st.info("Preprocessing has not run yet.")

with tab_models:
    st.subheader("Model Training and Comparison")
    st.write("Baseline/easier models only for now. Architecture is registry-based to add heavy models later.")

    if "clean_df" not in st.session_state:
        st.warning("Run preprocessing first from the Preprocessing tab.")
    else:
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.3, step=0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=999, value=0, step=1)

        if st.button("Train models", type="primary"):
            summary_df, details, trained_models = train_models(st.session_state["clean_df"], test_size, int(random_state))
            st.session_state["model_summary"] = summary_df
            st.session_state["model_details"] = details
            st.session_state["trained_models"] = trained_models
            st.success("Training complete.")

        if "model_summary" in st.session_state:
            st.dataframe(st.session_state["model_summary"], width="stretch")

            fig_scores = px.bar(
                st.session_state["model_summary"],
                x="Test Accuracy",
                y="Model",
                color="Test Accuracy",
                orientation="h",
                title="Model Comparison",
            )
            st.plotly_chart(fig_scores, width="stretch")

            selected_model = st.selectbox(
                "View detailed metrics for model",
                options=st.session_state["model_summary"]["Model"].tolist(),
            )
            model_detail = st.session_state["model_details"][selected_model]
            st.write("Confusion Matrix")
            st.write(model_detail["confusion_matrix"])
            st.text("Classification Report")
            st.text(model_detail["classification_report"])

with tab_predict:
    st.subheader("🔬 Patient Predictor")

    if "trained_models" not in st.session_state:
        st.warning("Please train models first in the Models tab")
    elif "label_encoders" not in st.session_state:
        st.warning("Please run preprocessing first in the Preprocessing tab")
    else:
        label_encoders = st.session_state["label_encoders"]
        trained_models = st.session_state["trained_models"]

        with st.form("patient_predictor_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                age = st.number_input("age", value=48.0)
                blood_pressure = st.number_input("blood_pressure", value=80.0)
                specific_gravity = st.number_input("specific_gravity", value=1.02, format="%.3f")
                albumin = st.number_input("albumin", value=1.0)
                sugar = st.number_input("sugar", value=0.0)
                blood_glucose_random = st.number_input("blood_glucose_random", value=121.0)
                blood_urea = st.number_input("blood_urea", value=36.0)
                serum_creatinine = st.number_input("serum_creatinine", value=1.2, format="%.3f")

            with c2:
                sodium = st.number_input("sodium", value=138.0)
                potassium = st.number_input("potassium", value=4.5, format="%.3f")
                haemoglobin = st.number_input("haemoglobin", value=13.0, format="%.3f")
                packed_cell_volume = st.number_input("packed_cell_volume", value=40.0)
                white_blood_cell_count = st.number_input("white_blood_cell_count", value=7800.0)
                red_blood_cell_count = st.number_input("red_blood_cell_count", value=4.8, format="%.3f")
                red_blood_cells = st.selectbox("red_blood_cells", options=["normal", "abnormal"])
                pus_cell = st.selectbox("pus_cell", options=["normal", "abnormal"])

            with c3:
                pus_cell_clumps = st.selectbox("pus_cell_clumps", options=["present", "notpresent"])
                bacteria = st.selectbox("bacteria", options=["present", "notpresent"])
                hypertension = st.selectbox("hypertension", options=["yes", "no"])
                diabetes_mellitus = st.selectbox("diabetes_mellitus", options=["yes", "no"])
                coronary_artery_disease = st.selectbox("coronary_artery_disease", options=["yes", "no"])
                appetite = st.selectbox("appetite", options=["good", "poor"])
                peda_edema = st.selectbox("peda_edema", options=["yes", "no"])
                aanemia = st.selectbox("aanemia", options=["yes", "no"])

            predict_clicked = st.form_submit_button("🔍 Predict", type="primary")

        if predict_clicked:
            input_row = {
                "age": age,
                "blood_pressure": blood_pressure,
                "specific_gravity": specific_gravity,
                "albumin": albumin,
                "sugar": sugar,
                "red_blood_cells": red_blood_cells,
                "pus_cell": pus_cell,
                "pus_cell_clumps": pus_cell_clumps,
                "bacteria": bacteria,
                "blood_glucose_random": blood_glucose_random,
                "blood_urea": blood_urea,
                "serum_creatinine": serum_creatinine,
                "sodium": sodium,
                "potassium": potassium,
                "haemoglobin": haemoglobin,
                "packed_cell_volume": packed_cell_volume,
                "white_blood_cell_count": white_blood_cell_count,
                "red_blood_cell_count": red_blood_cell_count,
                "hypertension": hypertension,
                "diabetes_mellitus": diabetes_mellitus,
                "coronary_artery_disease": coronary_artery_disease,
                "appetite": appetite,
                "peda_edema": peda_edema,
                "aanemia": aanemia,
            }

            input_df = pd.DataFrame([input_row])

            categorical_cols = [
                "red_blood_cells",
                "pus_cell",
                "pus_cell_clumps",
                "bacteria",
                "hypertension",
                "diabetes_mellitus",
                "coronary_artery_disease",
                "appetite",
                "peda_edema",
                "aanemia",
            ]

            for col in categorical_cols:
                if col in input_df.columns and col in st.session_state["label_encoders"]:
                    encoder = st.session_state["label_encoders"][col]
                    try:
                        encoded_val = int(encoder.transform([str(input_df[col].iloc[0])])[0])
                    except ValueError:
                        encoded_val = 0
                    input_df[col] = encoded_val

            # Force every column to numeric explicitly, column by column
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

            input_df = input_df.fillna(0)
            input_df = input_df.infer_objects()

            model_votes = {}
            for model_name, model in trained_models.items():
                pred = int(model.predict(input_df)[0])
                model_votes[model_name] = pred

            vote_series = pd.Series(model_votes)
            majority_pred = int(vote_series.mode().iloc[0])
            agree_count = int((vote_series == majority_pred).sum())
            confidence = (agree_count / len(vote_series)) * 100

            if majority_pred == 0:
                st.error(f"CKD detected (majority vote). Confidence: {confidence:.1f}%")
            else:
                st.success(f"Not CKD detected (majority vote). Confidence: {confidence:.1f}%")

            vote_df = pd.DataFrame(
                {
                    "Model": list(model_votes.keys()),
                    "Vote": ["✅ Not CKD" if pred == 1 else "🔴 CKD" for pred in model_votes.values()],
                }
            )
            st.dataframe(vote_df, width="stretch")

st.caption("Next phase ready: add patient-input form and per-model CKD prediction.")
