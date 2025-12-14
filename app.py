import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA App", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (5, 3)

st.title("Aplicatie EDA cu Streamlit")

# =========================
# CERINTA 1 – Incarcare si filtrare
# =========================

st.header("1. Încărcare și filtrare dataset")

uploaded_file = st.file_uploader("Încarcă un fișier CSV sau Excel", type=["csv", "xlsx"])

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception:
        return None

if uploaded_file:
    df = load_data(uploaded_file)

    if df is None:
        st.error("Eroare la citirea fișierului!")
    else:
        st.success("Fișier încărcat cu succes!")
        st.subheader("Primele 10 rânduri")
        st.dataframe(df.head(10))

        # =========================
        # ETAPA 2 – Curățarea datelor
        # =========================

        st.header("1.2. Curățarea datelor")

        st.subheader("Valori lipsă – înainte de curățare")
        st.write(df.isnull().sum())

        df_clean = df.copy()

        # completare valori lipsă
        for col in df_clean.select_dtypes(include=np.number).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        for col in df_clean.select_dtypes(exclude=np.number).columns:
            df_clean[col] = df_clean[col].fillna("Necunoscut")

        st.subheader("Valori lipsă – după curățare")
        st.write(df_clean.isnull().sum())

        st.subheader("Preview date curățate")
        st.dataframe(df_clean.head(10))

        # =========================
        # ETAPA 3 – Detectarea valorilor anormale (IQR)
        # =========================

        st.header("1.3. Detectarea valorilor anormale")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        outlier_summary = {}

        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
            outlier_summary[col] = {
                "Număr outlieri": len(outliers),
                "Procent": round(len(outliers) / len(df_clean) * 100, 2)
            }

        st.dataframe(pd.DataFrame(outlier_summary).T)

        # =========================
        # ETAPA 4 – Prelucrarea șirurilor de caractere
        # =========================

        st.header("1.4. Prelucrarea șirurilor de caractere")

        cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()

        if cat_cols:
            text_col = st.selectbox("Selectează coloană text", cat_cols)

            df_clean[text_col] = (
                df_clean[text_col]
                .str.lower()
                .str.strip()
                .str.replace(" ", "_", regex=False)
            )

            st.subheader("Date după procesare text")
            st.dataframe(df_clean[[text_col]].head(10))

        # =========================
        # ETAPA 5 – Standardizare și normalizare
        # =========================

        st.header("1.5. Standardizare și normalizare")

        if numeric_cols:
            scale_col = st.selectbox("Selectează coloană numerică", numeric_cols)

            scaler = StandardScaler()
            df_clean[scale_col + "_standardizat"] = scaler.fit_transform(
                df_clean[[scale_col]]
            )

            df_clean[scale_col + "_normalizat"] = (
                df_clean[scale_col] - df_clean[scale_col].min()
            ) / (
                df_clean[scale_col].max() - df_clean[scale_col].min()
            )

            st.subheader("Rezultat standardizare & normalizare")
            st.dataframe(df_clean[[scale_col, scale_col + "_standardizat", scale_col + "_normalizat"]].head(10))

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.subheader("Filtrare date")
        filtered_df = df.copy()

        for col in numeric_cols:
            min_val, max_val = st.slider(
                f"Interval pentru {col}",
                float(df[col].min()), float(df[col].max()),
                (float(df[col].min()), float(df[col].max()))
            )
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        for col in cat_cols:
            options = st.multiselect(f"Valori pentru {col}", df[col].dropna().unique())
            if options:
                filtered_df = filtered_df[filtered_df[col].isin(options)]

        st.write(f"Rânduri inițiale: {len(df)}")
        st.write(f"Rânduri după filtrare: {len(filtered_df)}")
        st.dataframe(filtered_df)

        # =========================
        # CERINTA 2 – Info despre dataset
        # =========================

        st.header("2. Informații generale despre dataset")
        st.write("Dimensiune:", df.shape)
        st.write("Tipuri de date:")
        st.write(df.dtypes)

        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        st.subheader("Valori lipsă")
        st.write(pd.DataFrame({"Missing": missing, "%": missing_pct}))

        fig, ax = plt.subplots(figsize=(4,3))
        missing.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title("Valori lipsă pe coloană")
        st.pyplot(fig)

        st.subheader("Statistici descriptive")
        st.write(df.describe().T)

        # =========================
        # CERINTA 3 – Histogramă & Boxplot
        # =========================

        st.header("3. Analiza unei coloane numerice")

        if numeric_cols:
            selected_num = st.selectbox("Alege o coloană numerică", numeric_cols)
            bins = st.slider("Număr de bins", 10, 100, 30)

            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df[[selected_num]].dropna()).flatten()

            fig1, ax1 = plt.subplots(figsize=(4,3))
            ax1.hist(scaled_values, bins=bins, color='steelblue', label='Distribuție')
            ax1.set_title(f"Histogramă – {selected_num}")
            ax1.legend()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(4,3))
            sns.boxplot(x=df[selected_num], ax=ax2, color='lightcoral')
            ax2.set_title(f"Boxplot – {selected_num}")
            st.pyplot(fig2)

            st.write("Medie:", df[selected_num].mean())
            st.write("Mediană:", df[selected_num].median())
            st.write("Deviație standard:", df[selected_num].std())

        # =========================
        # CERINTA 4 – Categorice
        # =========================

        st.header("4. Analiza coloanelor categorice")

        if cat_cols:
            selected_cat = st.selectbox("Alege o coloană categorică", cat_cols)
            freq = df[selected_cat].value_counts()
            freq_pct = df[selected_cat].value_counts(normalize=True) * 100
            st.write(pd.DataFrame({"Frecvență": freq, "%": freq_pct}))

            fig3, ax3 = plt.subplots(figsize=(4,3))
            sns.countplot(x=df[selected_cat], ax=ax3, palette='Set2')
            ax3.set_title(f"Frecvențe – {selected_cat}")
            plt.xticks(rotation=45)
            st.pyplot(fig3)

        # =========================
        # CERINTA 5 – Corelații & Outlieri
        # =========================

        st.header("5. Corelații și Outlieri")

        if numeric_cols:
            st.subheader("Matrice de corelație")
            corr = df[numeric_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(5,4))
            sns.heatmap(corr, annot=True, cmap='viridis', ax=ax4, cbar_kws={'label': 'Coeficient'})
            st.pyplot(fig4)

            st.subheader("Scatter plot")
            col_x = st.selectbox("Variabilă X", numeric_cols)
            col_y = st.selectbox("Variabilă Y", numeric_cols)

            fig5, ax5 = plt.subplots(figsize=(4,3))
            sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax5, color='purple', label='Observații')
            ax5.legend()
            st.pyplot(fig5)

            st.write("Coeficient Pearson:", df[col_x].corr(df[col_y]))

            st.subheader("Detectare outlieri (IQR)")
            outlier_info = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                outlier_info[col] = {
                    "count": len(outliers),
                    "percentage": (len(outliers) / len(df)) * 100
                }

            st.write(pd.DataFrame(outlier_info).T)

            for col in numeric_cols:
                fig6, ax6 = plt.subplots(figsize=(4,3))
                sns.boxplot(x=df[col], ax=ax6, color='orange')
                ax6.set_title(f"Outlieri – {col}")
                st.pyplot(fig6)
