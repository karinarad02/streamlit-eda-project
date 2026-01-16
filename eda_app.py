import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (5, 3)

def run():
    st.title("📊 Aplicație EDA cu Streamlit")

    uploaded_file = st.file_uploader("Încarcă un fișier CSV sau Excel", type=["csv", "xlsx"])

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
    # TABURI principale
    # =========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Curățarea datelor",
        "Detectare outlieri",
        "Prelucrare text",
        "Standardizare & normalizare"
    ])

    # -------------------------
    # TAB 1: Curățarea datelor
    # -------------------------
    with tab1:
        for col in df_clean.select_dtypes(include=np.number):
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        for col in df_clean.select_dtypes(exclude=np.number):
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        missing_df = pd.DataFrame({
            "Lipsă înainte": df.isnull().sum(),
            "Lipsă după": df_clean.isnull().sum()
        })
        st.subheader("Valori lipsă")
        st.dataframe(missing_df)
        st.subheader("Preview date curățate")
        st.dataframe(df_clean.head(10))

    # -------------------------
    # TAB 2: Detectare outlieri
    # -------------------------
    with tab2:
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        outlier_summary = {}

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
            outlier_summary[col] = {
                "Număr outlieri corectați": int(count),
                "Procent (%)": round(count / len(df_clean) * 100, 2),
                "Limită inferioară": round(lower,2),
                "Limită superioară": round(upper,2)
            }

        st.dataframe(pd.DataFrame(outlier_summary).T)

    # -------------------------
    # TAB 3: Prelucrare text + Label Encoding
    # -------------------------
    with tab3:
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
        if len(cat_cols) > 0:
            text_col = st.selectbox("Selectează coloană text", cat_cols)
            df_clean[text_col] = df_clean[text_col].str.lower().str.strip().str.replace(" ", "_", regex=False)
            le = LabelEncoder()
            df_clean[text_col + "_encoded"] = le.fit_transform(df_clean[text_col])
            st.subheader("Text + Label Encoding")
            st.dataframe(df_clean[[text_col, text_col + "_encoded"]].head(10))
        else:
            st.info("Nu există coloane categorice.")

    # -------------------------
    # TAB 4: Standardizare & Normalizare
    # -------------------------
    with tab4:
        if numeric_cols:
            scale_col = st.selectbox("Selectează coloană numerică", numeric_cols)
            scaler = StandardScaler()
            df_clean[scale_col + "_standardizat"] = scaler.fit_transform(df_clean[[scale_col]])
            df_clean[scale_col + "_normalizat"] = (df_clean[scale_col] - df_clean[scale_col].min()) / (df_clean[scale_col].max() - df_clean[scale_col].min())
            st.subheader("Rezultat")
            st.dataframe(df_clean[[scale_col, scale_col + "_standardizat", scale_col + "_normalizat"]].head(10))

    # -------------------------
    # Salvare pentru ML
    # -------------------------
    st.session_state["data"] = df_clean
    st.success("Datasetul curățat a fost salvat pentru modulul Machine Learning.")

    # -------------------------
    # Restul EDA: filtre, histograme, corelații
    # -------------------------
    # Poți să pui aici codul complet cu:
    # - filtrare numerică și categorică
    # - statistici descriptive
    # - histograme, boxplots
    # - matrice de corelație, scatter plots, outlieri
    # Exact ca în codul tău original, doar încadrat în `run()`
