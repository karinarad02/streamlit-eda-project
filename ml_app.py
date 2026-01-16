import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


st.title("🤖 Modul Machine Learning")

# ========================
# Verificare date din EDA
# ========================
if "data" not in st.session_state:
    st.warning("Mai întâi încarcă și curăță datele în modulul EDA.")
    st.stop()

df = st.session_state["data"]

st.subheader("Dataset folosit")
st.dataframe(df.head())

# ========================
# Problem setup
# ========================
target = st.selectbox("Selectează target", df.columns)

features = st.multiselect(
    "Selectează feature-uri",
    [c for c in df.columns if c != target],
    default=[c for c in df.columns if c != target]
)

X = df[features]
y = df[target]

problem_type = "classification" if y.nunique() < 20 else "regression"
st.info(f"Problemă detectată: {problem_type.upper()}")

# ========================
# Preprocessing
# ========================
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

impute_strategy = st.selectbox(
    "Imputare numerică",
    ["mean", "median"]
)

scaler_option = st.selectbox(
    "Scalare",
    ["None", "StandardScaler", "MinMaxScaler"]
)

num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
if scaler_option == "StandardScaler":
    num_steps.append(("scaler", StandardScaler()))
elif scaler_option == "MinMaxScaler":
    num_steps.append(("scaler", MinMaxScaler()))

num_pipeline = Pipeline(num_steps)

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_features),
    ("cat", cat_pipeline, categorical_features)
])

# ========================
# Split
# ========================
test_size = st.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.number_input("Random state", 0, 9999, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# ========================
# Models
# ========================
if problem_type == "classification":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(probability=True)
    }
else:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "SVR": SVR()
    }

selected_models = st.multiselect(
    "Selectează modele",
    list(models.keys())
)

# ========================
# Train & evaluate
# ========================
if st.button("🚀 Train models") and selected_models:
    results = []

    for name in selected_models:
        model = models[name]

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        st.subheader(name)

        if problem_type == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1": f1
            })

            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

        else:
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)

            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

    st.header("📊 Comparare modele")
    st.dataframe(pd.DataFrame(results))
