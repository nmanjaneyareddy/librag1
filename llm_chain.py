import os
import streamlit as st
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DEFAULT_HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def _get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def _make_llm():
    hf_token = (
        _get_secret("HUGGINGFACEHUB_API_TOKEN")
        or _get_secret("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
    )

    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Add HUGGINGFACEHUB_API_TOKEN or HF_TOKEN "
            "to Streamlit secrets or environment variables."
        )

    return HuggingFaceEndpoint(
        repo_id=DEFAULT_HF_MODEL,
        huggingfacehub_api_token=hf_token,
        temperature=0.2,
        max_new_tokens=512,
    )

class SimpleRetrievalQA:
    def __init__(self, retriever, llm, k=4):
        self.retriever = retriever
        self.llm = llm
        self.k = k

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the context to answer clearly and concisely. "
                "If the answer is not in the context, say you do not know.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs.get("input") or inputs.get("question") or inputs.get("query")

        if not question:
            return {"answer": ""}

        docs = self.retriever.invoke(question)
        docs = docs[: self.k]

        context_parts: List[str] = [
            getattr(doc, "page_content", "") for doc in docs
        ]

        context = "\n\n".join([part for part in context_parts if part])

        prompt_text = self.prompt.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt_text)

        if isinstance(response, str):
            answer = response
        else:
            answer = getattr(response, "content", str(response))

        return {"answer": answer}

def setup_qa_chain(vectorstore, k=4):
    if vectorstore is None:
        raise ValueError("vectorstore cannot be None")

    llm = _make_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return SimpleRetrievalQA(
        retriever=retriever,
        llm=llm,
        k=k
    )
