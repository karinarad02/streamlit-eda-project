import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


def run():
    st.title("🤖 Modul Machine Learning")

    # =========================
    # Verificare date
    # =========================
    if "data" not in st.session_state:
        st.warning("Mai întâi încarcă și curăță datele în modulul EDA.")
        st.stop()

    df = st.session_state["data"]

    st.subheader("Dataset folosit")
    st.dataframe(df.head())

    # ======================================================
    # PARTEA 1 — PROBLEM SETUP
    # ======================================================
    st.header("1. Problem Setup")

    target = st.selectbox("Selectează coloana țintă (target)", df.columns)

    features = st.multiselect(
        "Selectează feature-uri",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target]
    )

    X = df[features]
    y = df[target]

    problem_type = "classification" if y.nunique() < 20 else "regression"
    st.info(f"Tip problemă detectat automat: **{problem_type.upper()}**")

    # ======================================================
    # PARTEA 2 — PREPROCESARE
    # ======================================================
    st.header("2. Preprocesare")

    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns

    st.subheader("Opțiuni preprocesare")

    impute_strategy = st.selectbox(
        "Imputare numerică",
        ["mean", "median", "most_frequent"]
    )

    scaler_option = st.selectbox(
        "Scalare numerică",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    use_feature_selection = st.checkbox("Aplică SelectKBest (feature selection)")
    k_features = st.slider(
        "Număr features păstrate",
        1, max(1, X.shape[1]), min(10, X.shape[1])
    ) if use_feature_selection else None

    # Pipeline numeric
    num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if scaler_option == "StandardScaler":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler_option == "MinMaxScaler":
        num_steps.append(("scaler", MinMaxScaler()))

    num_pipeline = Pipeline(num_steps)

    # Pipeline categoric
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # ======================================================
    # PARTEA 3 — TRAIN / TEST SPLIT
    # ======================================================
    st.header("3. Train / Test Split")

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    # ======================================================
    # PARTEA 4 — MODELE & HIPERPARAMETRI
    # ======================================================
    st.header("4. Modele")

    models = {}

    if problem_type == "classification":
        st.subheader("Clasificare")

        if st.checkbox("Logistic Regression"):
            C = st.slider("LR: C", 0.01, 10.0, 1.0)
            models["Logistic Regression"] = LogisticRegression(
                C=C, max_iter=1000
            )

        if st.checkbox("Random Forest Classifier"):
            n_estimators = st.slider("RF: n_estimators", 50, 300, 100)
            max_depth = st.slider("RF: max_depth", 2, 20, 10)
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

        if st.checkbox("SVM (SVC)"):
            C = st.slider("SVC: C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("SVC kernel", ["rbf", "linear"])
            models["SVM"] = SVC(C=C, kernel=kernel, probability=True)

    else:
        st.subheader("Regresie")

        if st.checkbox("Linear Regression"):
            models["Linear Regression"] = LinearRegression()

        if st.checkbox("Random Forest Regressor"):
            n_estimators = st.slider("RF: n_estimators", 50, 300, 100)
            max_depth = st.slider("RF: max_depth", 2, 20, 10)
            models["Random Forest"] = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

        if st.checkbox("SVR"):
            C = st.slider("SVR: C", 0.1, 10.0, 1.0)
            models["SVR"] = SVR(C=C)

    # ======================================================
    # PARTEA 5 — ANTRENARE & EVALUARE
    # ======================================================
    st.header("5. Evaluare & Comparare")

    metric_select = st.selectbox(
        "Metrică pentru best model",
        ["Accuracy", "F1"] if problem_type == "classification"
        else ["RMSE", "R2"]
    )

    if st.button("🚀 Train models") and models:
        results = []

        for name, model in models.items():
            steps = [("preprocessor", preprocessor)]

            if use_feature_selection:
                selector = SelectKBest(
                    score_func=f_classif if problem_type == "classification" else f_regression,
                    k=k_features
                )
                steps.append(("selector", selector))

            steps.append(("model", model))

            pipeline = Pipeline(steps)

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            st.subheader(name)

            if problem_type == "classification":
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, average="weighted")
                rec = recall_score(y_test, preds, average="weighted")
                f1 = f1_score(y_test, preds, average="weighted")

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1
                })

                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                st.pyplot(fig)

            else:
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                results.append({
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

        results_df = pd.DataFrame(results)

        st.subheader("Comparare modele")
        st.dataframe(results_df)

        best_model = (
            results_df.sort_values(metric_select, ascending=False).iloc[0]
            if metric_select != "RMSE"
            else results_df.sort_values(metric_select).iloc[0]
        )

        st.success(f"🏆 Best model: **{best_model['Model']}**")
