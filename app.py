"""Streamlit chat UI for Bring Books to Life.

Run:  streamlit run app.py
Needs GROQ_API_KEY in the environment / .env (free Groq key).
"""
import streamlit as st

from agent import BookChat
from personas import PERSONAS, DEFAULT

st.set_page_config(page_title="Bring Books to Life", page_icon="🎭", layout="centered")
st.title("🎭 Bring Books to Life")
st.caption("Chat with a character from a book. They stay in character, answer from the text, and remember you.")

with st.sidebar:
    key = st.selectbox("Character", list(PERSONAS), format_func=lambda k: PERSONAS[k].name,
                       index=list(PERSONAS).index(DEFAULT))
    uploaded = st.file_uploader("Ground this character in your own source (.txt) — keeps their persona", type=["txt"])
    if st.button("Start / reset conversation", type="primary"):
        src = uploaded.read().decode("utf-8", "ignore") if uploaded else None
        st.session_state.chat = BookChat(key, source_text=src)
        st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = BookChat(DEFAULT)
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if prompt := st.chat_input(f"Say something to {st.session_state.chat.persona.name}…"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking in character…"):
            reply = st.session_state.chat.say(prompt)
        st.write(reply)
    st.session_state.history.append(("assistant", reply))
