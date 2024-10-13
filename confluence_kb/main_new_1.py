import re
import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
import requests
import pandas as pd
import urllib.parse
import streamlit as st
from dotenv import load_dotenv
import pickle
from pydantic import Field
from PyPDF2 import PdfReader
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain, chain
from langchain.callbacks import get_openai_callback
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import ConfluenceLoader
from core.files.confluence_store import CustomVectorStore
from config import Config
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts import PromptTemplate

config = Config()
llama_config = config.get_api_llama_config()

prompt_template = PromptTemplate(
    input_variables=["question", "context", "example"],
    template="""
    You are an expert in analyzing content and providing detailed, relevant information only from the following context given.

    User question: {question}

    The following context contains content extracted from various pages. Carefully analyze the content and provide information that is most relevant to the user's question.

    context:{context}

    source:{example}

    Please now generate a detailed response to the user question based on the above context only, and be sure to provide the source link.

    source:
    """
)

class OllamaLLM(LLM):
    model_name: str = Field(default="llama3.1", description="Name of the model to use in Ollama API")
    api_url: str = Field(default="http://localhost:11434/api/generate", description="API endpoint for Ollama LLM")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3.1",
            "prompt": prompt,
            "max_tokens": 5000,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        response_placeholder = st.empty()
        response = requests.post(self.api_url, json=data, stream=True)

        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    chunk_data = chunk.decode("utf-8")
                    try:
                        chunk_json = json.loads(chunk_data)
                        token = chunk_json.get("response", "")
                        full_response += token
                        response_placeholder.info(f"`Answer:`\n\n{full_response}")
                    except json.JSONDecodeError:
                        print("Failed to decode JSON from LLaMA API")
                        break
            return full_response
        else:
            return f"Error querying LLaMA API: {response.status_code}"


    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "ollama"
    


with st.sidebar:
    st.markdown('''
    ## About:
    **Revolutionizing Legal Workflows with Intelligent Document Analysis**

    **:blue[Some prominent prompts]** 
     - Explain key components of Central Data Hub
    ''')

    st.markdown(
        """
        <style>
        .bottom-left-text {
            position: fixed;
            bottom: 10px;
            left: 10px;
            font-size: 14px;
            color: #333;  /* Customize the text color */
            z-index: 100;
        }
        </style>
        <div class="bottom-left-text">
            **POWERED BY LLM** 
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .dataframe th, .dataframe td {
            max-width: 150px;
            word-wrap: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

load_dotenv()


def convert_page_name_to_file_name(page_name):
    file_name = page_name.replace(" ", "_")
    file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)
    return file_name


def query_llama_api(prompt):
    llama_api_url = llama_config['api_endpoint']
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "max_tokens": 5000,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response_placeholder = st.empty()

    response = requests.post(llama_api_url, json=data, stream=True)

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

def convert_to_tags(question):
    words = re.findall(r'\b\w+\b', question.lower())
    
    stop_words = {'the', 'is', 'in', 'at', 'which', 'on', 'a', 'an', 'and', 'for', 'to', 'of', 'by', 'with'}
    
    tags = [word for word in words if word not in stop_words]
    
    return ', '.join(f'"{tag}"' for tag in tags)

def retrieve_and_generate_response_from_all_stores(question: str, retrievers: Dict[str, VectorStoreRetriever], api_url: str, model_name: str, top_k: int = 10) -> str:    
    context_explanation = ""
    examples=""
    for store_name, retriever in retrievers.items():
        retrieved_documents = retriever.invoke(question, top_k=top_k)
        context_explanation += f"Content from {store_name}:\n"
        for i, doc in enumerate(retrieved_documents):
            source = doc.metadata.get('source', 'Not Available') 
            context_explanation += f"Document {i+1}: {doc.page_content} Source: {source}\n"
            examples = f" source :  {source}\n"
            print(context_explanation)
    populated_prompt = prompt_template.format(
        question=question,
        context=context_explanation,
        example=examples
    )
    ollama_llm = OllamaLLM(model_name="llama3.1", api_url=llama_config['api_endpoint'])  

    response = ollama_llm._call(prompt=populated_prompt)

    return response

def main():
    global applicationDatasourceDocsMemory
    st.header("Knowledge base (Confluence)")
      
    question = st.text_input("Ask AI")

    if question:
        vector_stores, metadata_store = st.session_state["vector_stores"], st.session_state["metadata_store"]
        retrievers = {}
        for key, _vectorstore in vector_stores.items():
            retrievers[key] = VectorStoreRetriever(
                                    vectorstore=_vectorstore,
                                    metadata=metadata_store,
                                    search_type="similarity")

        retrieve_and_generate_response_from_all_stores(
            question=question,
            retrievers=retrievers,
            api_url=llama_config["api_endpoint"],
            model_name="llama3.1")

def run():
    main()