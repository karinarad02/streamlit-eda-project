import streamlit as st
import eda_app
import ml_app

st.set_page_config(page_title="EDA + ML App", layout="wide")

st.sidebar.title("Navigare")
page = st.sidebar.radio(
    "Selectează modul:",
    ["📊 Explorare date (EDA)", "🤖 Machine Learning"]
)

if page == "📊 Explorare date (EDA)":
    eda_app.run()

elif page == "🤖 Machine Learning":
    ml_app.run()
