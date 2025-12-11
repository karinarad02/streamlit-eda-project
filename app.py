import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA App", layout="wide")
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

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.subheader("Filtrare date")

        filtered_df = df.copy()

        # Numeric filters
        for col in numeric_cols:
            min_val, max_val = st.slider(
                f"Interval pentru {col}",
                float(df[col].min()), float(df[col].max()),
                (float(df[col].min()), float(df[col].max()))
            )
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        # Categorical filters
        for col in cat_cols:
            options = st.multiselect(f"Valori pentru {col}", df[col].unique())
            if options:
                filtered_df = filtered_df[filtered_df[col].isin(options)]

        st.write(f"Rânduri inițiale: {len(df)}")
        st.write(f"Rânduri după filtrare: {len(filtered_df)}")
        st.dataframe(filtered_df)

        # =========================
        # CERINTA 2 – Info despre dataset
        # =========================

        st.header("2. Informații generale despre dataset")

        st.subheader("Număr rânduri și coloane")
        st.write(df.shape)

        st.subheader("Tipuri de date")
        st.write(df.dtypes)

        st.subheader("Valori lipsă")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        st.write(pd.DataFrame({"Missing": missing, "%": missing_pct}))

        st.subheader("Grafic valori lipsă")
        fig, ax = plt.subplots()
        missing.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.subheader("Statistici descriptive pentru coloane numerice")
        st.write(df.describe().T)

        # =========================
        # CERINTA 3 – Histogramă & Boxplot
        # =========================

        st.header("3. Analiza unei coloane numerice")

        if numeric_cols:
            selected_num = st.selectbox("Alege o coloană numerică", numeric_cols)
            bins = st.slider("Număr de bins", 10, 100, 30)

            fig1, ax1 = plt.subplots()
            ax1.hist(df[selected_num].dropna(), bins=bins)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[selected_num], ax=ax2)
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

            fig3, ax3 = plt.subplots()
            sns.countplot(x=df[selected_cat], ax=ax3)
            plt.xticks(rotation=45)
            st.pyplot(fig3)

        # =========================
        # CERINTA 5 – Corelații & Outlieri
        # =========================

        st.header("5. Corelații și Outlieri")

        if numeric_cols:
            # Heatmap
            st.subheader("Matrice de corelație")
            corr = df[numeric_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="viridis", ax=ax4)
            st.pyplot(fig4)

            # Scatter plot
            st.subheader("Scatter plot între două variabile")
            col_x = st.selectbox("Variabilă X", numeric_cols)
            col_y = st.selectbox("Variabilă Y", numeric_cols)

            fig5, ax5 = plt.subplots()
            sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax5)
            st.pyplot(fig5)

            st.write("Coeficient Pearson:", df[col_x].corr(df[col_y]))

            # Outliers IQR
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

            st.subheader("Vizualizare outlieri")

            for col in numeric_cols:
                fig6, ax6 = plt.subplots()
                sns.boxplot(x=df[col], ax=ax6)
                ax6.set_title(f"Outlieri pentru {col}")
                st.pyplot(fig6)
