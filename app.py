"""University Policy Copilot - Streamlit UI.

Run:  streamlit run app.py
Pages: Chat (chat_page.py) and Trace dashboard (trace_dashboard.py).
Mode:  POLICY_AGENT_MODE=mock (default, sample data) | live (agent team code via ui_adapter.py)
"""
import os

import streamlit as st

st.set_page_config(page_title=os.environ.get("APP_TITLE", "University Policy Copilot"),
                   page_icon="📘", layout="wide")

page = st.navigation([
    st.Page("chat_page.py", title="Chat", icon=":material/chat:", default=True),
    st.Page("trace_dashboard.py", title="Trace dashboard", icon=":material/monitoring:"),
])
page.run()
