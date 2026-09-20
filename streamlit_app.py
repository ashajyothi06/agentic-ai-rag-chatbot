import os

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Agentic AI eBook RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI eBook – Grounded RAG Chatbot")
st.caption(
    "Answers are generated only from retrieved passages in the provided Agentic AI eBook."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the Agentic AI eBook...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and grounding..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()

                st.markdown(payload["answer"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{payload['confidence']:.0%}")
                c2.metric("Grounded", "Yes" if payload["grounded"] else "No")
                c3.metric("Model", payload["model"])

                with st.expander(
                    f"Retrieved context ({len(payload['context_chunks'])} chunks)"
                ):
                    for idx, chunk in enumerate(payload["context_chunks"], start=1):
                        page = chunk.get("page")
                        st.markdown(
                            f"**Chunk {idx} · Page {page or 'N/A'} · "
                            f"Similarity {chunk['score']:.3f}**"
                        )
                        st.write(chunk["content"])
                        st.divider()

                assistant_text = payload["answer"]

            except requests.RequestException as exc:
                assistant_text = (
                    "Could not reach the FastAPI backend. Start it with "
                    "`uvicorn app.main:app --reload`.\n\n"
                    f"Error: {exc}"
                )
                st.error(assistant_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )

with st.sidebar:
    st.header("About")
    st.write("Pipeline: PDF → chunks → embeddings → Pinecone → LangGraph → LLM")
    st.write("The API also returns retrieved chunks and a confidence score.")
    st.code(API_URL, language=None)
