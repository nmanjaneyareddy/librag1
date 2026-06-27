# app.py
import streamlit as st

from vectorstore import load_or_create_vector_store
from llm_chain import setup_qa_chain

st.set_page_config(
    page_title="IGIDR Library Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 IGIDR Library Assistant")
st.caption("Ask questions about the IGIDR library PDF. I can also return links if they are present in the document.")

with st.sidebar:
    st.header("Settings")

    k = st.slider(
        "Number of document chunks to search",
        min_value=2,
        max_value=8,
        value=4,
        step=1
    )

    st.markdown("### Tips")
    st.markdown(
        """
        Try questions like:

        - What are the library timings?
        - How can I access e-resources?
        - Give me useful links from the document.
        - What services does the library provide?
        """
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

try:
    vectorstore = load_or_create_vector_store()
    qa_chain = setup_qa_chain(vectorstore, k=k)
except Exception as exc:
    st.error("Startup failed: " + type(exc).__name__ + ": " + str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m your IGIDR Library Assistant. "
                "Ask me anything about the library PDF, including services, timings, rules, or useful links."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the IGIDR library...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            try:
                result = qa_chain.invoke({"input": question})

                if isinstance(result, dict):
                    answer = result.get("answer", "")
                else:
                    answer = str(result)

                if not answer:
                    answer = "I could not find an answer in the PDF."

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:
                error_message = "Query failed: " + type(exc).__name__ + ": " + str(exc)
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
