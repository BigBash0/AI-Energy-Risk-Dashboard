import streamlit as st


def apply_theme():

    st.markdown(
        """
        <style>

        .stApp {
            background: linear-gradient(
                180deg,
                #F7FBF7 0%,
                #EEF5F1 45%,
                #F8FAFC 100%
            );
        }

        h1,h2,h3 {
            color:#12372A;
        }

        </style>
        """,
        unsafe_allow_html=True
    )