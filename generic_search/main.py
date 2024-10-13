import requests
import streamlit as st
import json
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import Config
from core.llm.llm_invoker import ModelInvoker

config = Config()
llama_config = config.get_api_llama_config()
model_invoker = ModelInvoker()


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)

def run():
    st.title("Ask AI")

    model_name = st.selectbox("Choose a model", ["OPENAI", "llama3.1", "llama3.1:405b"])

    search_query = st.text_input("How can I assist you today? Ask your question here")

    if search_query:
        st.write(f"looking for: {search_query}")
        result = model_invoker.call_model(model_name,search_query,"system","You are SQL Expert")
    else:
        st.error("Please enter a search query.")    

