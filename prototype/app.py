"""
RelayGuard -- entry point
Run:  python -m streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="RelayGuard",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/dashboard.py",      title="Prototype",   icon="\u26a1", default=True),
    st.Page("pages/1_Demo_Guide.py",   title="Demo Guide",  icon="\U0001f4d6"),
])
pg.run()
