import streamlit as st

st.set_page_config(page_title="EDA + ML App", layout="wide")

st.sidebar.title("Navigare")
page = st.sidebar.radio(
    "Selectează modul:",
    ["📊 Explorare date (EDA)", "🤖 Machine Learning"]
)

if page == "📊 Explorare date (EDA)":
    import eda_app
    eda_app.run()

elif page == "🤖 Machine Learning":
    import ml_app
    ml_app.run()
