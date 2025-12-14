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
        # ETAPE DUPĂ ÎNCĂRCARE – TABURI
        # =========================

        tab1, tab2, tab3, tab4 = st.tabs([
            "1.1 Curățarea datelor",
            "1.2 Detectare outlieri",
            "1.3 Prelucrare text",
            "1.4 Standardizare & normalizare"
        ])

        df_clean = df.copy()

        # =========================
        # TAB 1 – Curățarea datelor
        # =========================
        with tab1:
            # Numerice → media
            for col in df_clean.select_dtypes(include=np.number).columns:
                mean_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_value)

            # Categorice → valoarea cea mai frecventă (moda)
            for col in df_clean.select_dtypes(exclude=np.number).columns:
                if not df_clean[col].mode().empty:
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_value)

            st.subheader("Număr valori lipsă (înainte de curățare)")
            missing_before = df.isnull().sum().reset_index()
            missing_before.columns = ['Coloană', 'Lipsă']
            st.dataframe(missing_before)

            st.subheader("Număr valori lipsă (după curățare)")
            missing_after = df_clean.isnull().sum().reset_index()
            missing_after.columns = ['Coloană', 'Lipsă']
            st.dataframe(missing_after)

            st.subheader("Preview date curățate")
            st.dataframe(df_clean.head(10))

        # =========================
        # TAB 2 – Detectare și corectare outlieri (IQR)
        # =========================
        with tab2:
            st.subheader("Detectare și corectare outlieri (metoda IQR)")

            numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
            outlier_summary = {}

            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                # Detectare outlieri
                outliers_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                outlier_count = outliers_mask.sum()

                # Corectare outlieri (winsorizare)
                df_clean.loc[df_clean[col] < lower, col] = lower
                df_clean.loc[df_clean[col] > upper, col] = upper

                outlier_summary[col] = {
                    "Număr outlieri corectați": int(outlier_count),
                    "Procent (%)": round(outlier_count / len(df_clean) * 100, 2),
                    "Limită inferioară": round(lower, 2),
                    "Limită superioară": round(upper, 2)
                }

            st.dataframe(pd.DataFrame(outlier_summary).T)

            st.success("✔ Outlierii au fost detectați și corectați folosind metoda IQR.")

        # =========================
        # TAB 3 – Prelucrare text + Label Encoding
        # =========================
        with tab3:
            cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()

            if cat_cols:
                text_col = st.selectbox("Selectează coloană text", cat_cols)

                # Prelucrare text
                df_clean[text_col] = (
                    df_clean[text_col]
                    .str.lower()
                    .str.strip()
                    .str.replace(" ", "_", regex=False)
                )

                # Label Encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_clean[text_col + "_encoded"] = le.fit_transform(df_clean[text_col])

                st.subheader("După prelucrare text + Label Encoding")
                st.dataframe(df_clean[[text_col, text_col + "_encoded"]].head(10))

            else:
                st.info("Nu există coloane categorice.")

        # =========================
        # TAB 4 – Standardizare & normalizare
        # =========================
        with tab4:
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

                st.subheader("Rezultat")
                st.dataframe(
                    df_clean[
                        [scale_col, scale_col + "_standardizat", scale_col + "_normalizat"]
                    ].head(10)
                )

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
        st.write(pd.DataFrame({"Lipsă": missing, "%": missing_pct}))

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
