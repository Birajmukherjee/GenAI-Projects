from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import json
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager  # Fixing CallbackManager import
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings    
from langchain.vectorstores import Chroma
import os
import requests
from config import Config
import streamlit as st

config = Config()
llama_config = config.get_api_llama_config()

n_gpu_layers = 1  
n_batch = 512  
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


def query_llama_api(prompt, user_info):
    llama_api_url = llama_config['api_endpoint']
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "max_tokens": 5000,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    headers = {
        "Authorization": f"Bearer {user_info['accessToken']}"
    }

    response_placeholder = st.empty()

    response = requests.post(llama_api_url, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk_data = chunk.decode("utf-8")
                chunk_json = json.loads(chunk_data)
                token = chunk_json.get("response", "")
                full_response += token

                response_placeholder.info(f"`Answer:`\n\n{full_response}")

                if chunk_json.get("done", False):
                    break
        return full_response
    else:
        return f"Error querying LLaMA API: {response.status_code}"

def run():
    #st.header("AI-Powered Knowledgebase")

    with st.sidebar:
        st.markdown('''
        ## Intelligent Knowledge Base ChatBot:
        **The Knowledge Base Bot ensures employees can access the information they need instantly, improving productivity and consistency.**
        
        **:blue[Some prominent prompts]** 
        - Please provide a summary of the key terms and conditions outlined in the terminal port concession document.
        - Analyze the financial implications of the concession agreement, including revenue-sharing models and cost structures.
        - List the compliance requirements that the concessionaire must adhere to as per the concession agreement.
        - What performance metrics are specified in the concession agreement to evaluate the concessionaire's effectiveness?
        - Outline the legal obligations of both parties as stated in the terminal port concession document.
        - Describe the legal provisions included in the concession agreement.
        ''')

    load_confluence_flag = False
    if load_confluence_flag:
        load_confluence_data(chromaVectorStore)

    query = st.text_input("Ask questions about your confluence pages and contents:")
    
    if query:
        if "user_info" in st.session_state:
            user_info = st.session_state["user_info"]
            query_llama_api(query, user_info) 
        else:
            st.error("User is not authenticated. Please provide an access token.")

def main():
    load_dotenv()
    run()

if __name__ == "__main__":
    main()
