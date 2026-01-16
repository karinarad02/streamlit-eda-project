import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (5, 3)


def run():
    st.title("📊 Aplicație EDA cu Streamlit")

    # =========================
    # CERINTA 1 – Incarcare
    # =========================
    st.header("1. Încărcare și filtrare dataset")

    uploaded_file = st.file_uploader(
        "Încarcă un fișier CSV sau Excel",
        type=["csv", "xlsx"]
    )

    def load_data(file):
        try:
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)
        except Exception:
            return None

    if uploaded_file is None:
        st.info("Încarcă un fișier pentru a continua.")
        return

    df = load_data(uploaded_file)

    if df is None:
        st.error("Eroare la citirea fișierului!")
        return

    st.success("Fișier încărcat cu succes!")
    st.subheader("Primele 10 rânduri")
    st.dataframe(df.head(10))

    df_clean = df.copy()

    # =========================
    # TABURI PRINCIPALE
    # =========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Curățarea datelor",
        "Detectare outlieri",
        "Prelucrare text",
        "Standardizare & normalizare"
    ])

    # =========================
    # TAB 1 – Curățare date
    # =========================
    with tab1:
        for col in df_clean.select_dtypes(include=np.number):
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        for col in df_clean.select_dtypes(exclude=np.number):
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        missing_df = pd.DataFrame({
            "Lipsă înainte": df.isnull().sum(),
            "Lipsă după": df_clean.isnull().sum()
        })

        st.subheader("Valori lipsă")
        st.dataframe(missing_df)
        st.dataframe(df_clean.head(10))

    # =========================
    # TAB 2 – Outlieri (IQR)
    # =========================
    with tab2:
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        summary = {}

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = (df_clean[col] < lower) | (df_clean[col] > upper)
            count = mask.sum()

            df_clean.loc[df_clean[col] < lower, col] = lower
            df_clean.loc[df_clean[col] > upper, col] = upper

            summary[col] = count

        st.subheader("Outlieri corectați")
        st.dataframe(pd.DataFrame.from_dict(
            summary, orient="index", columns=["Număr outlieri"]
        ))

    # =========================
    # TAB 3 – Text + Encoding
    # =========================
    with tab3:
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns

        if len(cat_cols) == 0:
            st.info("Nu există coloane categorice.")
        else:
            col = st.selectbox("Selectează coloană text", cat_cols)

            df_clean[col] = (
                df_clean[col]
                .str.lower()
                .str.strip()
                .str.replace(" ", "_", regex=False)
            )

            le = LabelEncoder()
            df_clean[col + "_encoded"] = le.fit_transform(df_clean[col])

            st.dataframe(df_clean[[col, col + "_encoded"]].head(10))

    # =========================
    # TAB 4 – Scalare
    # =========================
    with tab4:
        num_cols = df_clean.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            col = st.selectbox("Selectează coloană numerică", num_cols)
            scaler = StandardScaler()
            df_clean[col + "_standardizat"] = scaler.fit_transform(df_clean[[col]])

            df_clean[col + "_normalizat"] = (
                df_clean[col] - df_clean[col].min()
            ) / (
                df_clean[col].max() - df_clean[col].min()
            )

            st.dataframe(
                df_clean[[col, col + "_standardizat", col + "_normalizat"]].head()
            )

    # =========================
    # SALVARE PENTRU ML
    # =========================
    st.session_state["data"] = df_clean
    st.success("Datasetul curățat a fost salvat pentru modulul Machine Learning.")
