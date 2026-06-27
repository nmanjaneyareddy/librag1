# llm_chain.py
import streamlit as st
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def _make_llm():
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DEEPSEEK_API_KEY in Streamlit secrets")

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=512,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )


class SimpleRetrievalQA:
    """
    Minimal, stable Retrieval-QA implementation.
    No langchain.chains usage.
    """

    def __init__(self, retriever, llm, k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=(
                "You are a friendly and helpful IGIDR Library Assistant. "
                "Answer the user's question using only the context below. "
                "Be conversational, clear, and practical. "
                "If useful links or URLs are present in the context, include them exactly as written. "
                "If the answer is not available in the context, say that you could not find it in the PDF. "
                "Do not make up information.\n\n"
                "Context:\n{context}\n\n"
                "User question:\n{question}\n\n"
                "Assistant answer:"
            ),
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs.get("input") or inputs.get("question")
        if not question:
            return {"answer": ""}
        docs = self.retriever.invoke(question)[: self.k]

        context_parts: List[str] = []
        for d in docs:
            text = getattr(d, "page_content", None)
            if text:
                context_parts.append(text)

        context = "\n\n".join(context_parts)
        prompt_text = self.prompt.format(context=context, question=question)

        if hasattr(self.llm, "invoke"):
            answer = self.llm.invoke(prompt_text).content
        else:
            answer = self.llm.predict(prompt_text)

        return {"answer": answer}


def setup_qa_chain(vectorstore, k: int = 4):
    if vectorstore is None:
        raise ValueError("vectorstore cannot be None")

    llm = _make_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return SimpleRetrievalQA(retriever, llm, k=k)
