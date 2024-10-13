import io
import os
import json
from PIL import Image
import PyPDF2
from docx import Document
import pandas as pd
import urllib.parse
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from config import Config
import requests
from PyPDF2 import PdfReader
import pytesseract
import re
from core.llm.llm_invoker import ModelInvoker
from langchain.prompts import PromptTemplate

config = Config()
llama_config = config.get_api_llama_config()
model_invoker = ModelInvoker()

one_drive_metadata_store = {}

knowledgebase_prompt_template = PromptTemplate(
    input_variables=["input_documents", "metadata", "question"],
    template="""
    You are an expert assistant that provides detailed and structured responses. Based on the Content provided below, generate a clear, concise, and accurate response to the question. Your response should be organized into steps or categories with detailed descriptions.

    Ensure that you:
    - Understand the question carefully.
    - Generate an organized and structured response that directly addresses the user's question or statement.
    - Give illustrations or code examples to achive the task from the Content
    - Use clear titles or headings for each step or section.
    - Provide detailed explanations for each step or process.
    - Include links from the metadata under each step when relevant, like this: "(Source: [Document Link])".
    - If no link is available for a particular step, indicate "(Source: No link available)".

    Content:{input_documents}

    Metadata: {metadata}

    Question: {question}

    Now, based on the Content provided, generate a detailed and well-organized response that answers the question. Include relevant sources where applicable.
    """
)

local_store = "./one_drive_local_store"
os.makedirs(local_store, exist_ok=True)


config = Config()
llama_config = config.get_api_llama_config()

indexed_files = {}

def convert_page_name_to_file_name(page_name):
    file_name = page_name.replace(" ", "_")
    file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)
    return file_name

def extract_text_from_file(file_path):
    text = ""
    
    with open(file_path, 'rb') as f:
        file_content = f.read()

    if file_path.endswith('.pdf'):
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    
    elif file_path.endswith('.docx'):
        doc = Document(io.BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text
    
    elif file_path.endswith('.txt'):
        text = file_content.decode("utf-8")
    
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(io.BytesIO(file_content))
        text = df.to_string()
    
    elif file_path.endswith('.jpg') or file_path.endswith('.png'):
        img = Image.open(io.BytesIO(file_content))
        text = pytesseract.image_to_string(img)
    
    return text

load_dotenv()

def process_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    text = json.dumps(data, indent=2)
    return text

def download_files_from_onedrive(api_key, folder_id):
    global indexed_files, index, one_drive_metadata_store  # Declare one_drive_metadata_store as global to modify it inside the function
    headers = {"Authorization": f"Bearer {api_key}"}
        
    response = requests.get(f"https://graph.microsoft.com/v1.0/groups/0b394df8-c35d-4dc0-ad16-4bdf23036baa/drive/items/{folder_id}/children", headers=headers)
    
    if response.status_code == 200:
        files = response.json().get('value', [])
        for file in files:
            file_id = file['id']
            file_name = file['name']
            last_modified = file['lastModifiedDateTime']
            file_link = file['webUrl']  
            
            # Check if the file is new or has been updated
            if file_id not in indexed_files or indexed_files[file_id]['last_modified'] != last_modified:
                file_content_response = requests.get(f"https://graph.microsoft.com/v1.0/groups/0b394df8-c35d-4dc0-ad16-4bdf23036baa/drive/items/{file_id}/content", headers=headers)
                print(file_content_response)
                if file_content_response.status_code == 200:
                    file_content = file_content_response.content
                    
                    local_file_path = os.path.join(local_store, file_name)
                    with open(local_file_path, "wb") as local_file:
                        local_file.write(file_content)
                    
                    metadatas = [{
                        "fileId": file_id,
                        "fileName": file_name,
                        "lastModified": last_modified,
                        "url": file_link
                    }]
                    store_name = file_name[:-4] 

                    one_drive_metadata_store[convert_page_name_to_file_name(store_name)] = metadatas  # Now correctly storing metadata

                    print(f"Downloaded {file_name}")
                else:
                    print(f"Failed to download file content: {file_content_response.status_code}")
    else:
        print(f"Failed to list OneDrive folder contents: {response.status_code}")

def main():
    global one_drive_metadata_store  # Declare metadata_store as global to ensure it's accessed correctly
    one_drive_vector_stores = {}

    if "one_drive_vector_stores" not in st.session_state and "one_drive_metadata_store" not in st.session_state:
        if st.session_state["api_key"] and st.session_state["folder_id"]:
            download_files_from_onedrive(st.session_state["api_key"], st.session_state["folder_id"])
            directory_path = "./one_drive_local_store"

            if not os.path.exists(directory_path):
                print(f"The directory '{directory_path}' does not exist.")
                return

            files = [f for f in os.listdir(directory_path)]

            if not files:
                print("No files found in the directory.")
                return

            for file in files:
                file_path = os.path.join(directory_path, file)

                if not os.path.exists(file_path):
                    print(f"File {file} not found. Skipping.")
                    continue

                text = extract_text_from_file(file_path)

                if not text.strip():  
                    print(f"No content extracted from {file}. Skipping.")
                    continue
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                store_name = file[:-4]  # Remove the extension
                store_name = convert_page_name_to_file_name(store_name)
                print(f"Store name : {store_name}")
                if os.path.exists(f"{local_store}/{store_name}.pkl"):
                    with open(f"{local_store}/{store_name}.pkl", "rb") as f:
                        one_drive_vector_stores[store_name] = pickle.load(f)
                else:
                    embeddings = OpenAIEmbeddings() 
                    one_drive_vector_stores[store_name] = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{local_store}/{store_name}.pkl", "wb") as f:
                        pickle.dump(one_drive_vector_stores[store_name], f)

            with open(f"{local_store}/one_drive_metadata_store.pkl", "wb") as f:
                pickle.dump(one_drive_metadata_store, f)

            st.session_state["one_drive_vector_stores"] = one_drive_vector_stores
            st.session_state["one_drive_metadata_store"] = one_drive_metadata_store               
        else:
            st.warning("Please enter your API key and folder ID to start.")


    onedrive_question_input = st.text_input("What you love to know (OneDrive Knowledgebase):")

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

    if onedrive_question_input:

        if "one_drive_vector_stores" in st.session_state and "one_drive_metadata_store" in st.session_state:
            one_drive_vector_stores = st.session_state["one_drive_vector_stores"]
            one_drive_metadata_store = st.session_state["one_drive_metadata_store"]
            
            with open(f"{local_store}/one_drive_metadata_store.pkl", "rb") as f:
                one_drive_metadata_store = pickle.load(f)

            docs_with_sources = []
            relevant_sources = set()  

            for file, vector_store in one_drive_vector_stores.items():
                results = vector_store.similarity_search(query=onedrive_question_input, k=3)
                for result in results:
                    chunk_idx = result.metadata.get("chunk_index", 0)
                    result.metadata.update(one_drive_metadata_store[file][chunk_idx])
                    docs_with_sources.append(result)

                    relevant_sources.add(result.metadata.get("url", "Unknown source"))

            model_name=st.session_state['model_name']

            if model_name == "OpenAI":
                print(f"Powerd by OpenAI GPT-4o query={onedrive_question_input}")

                llm = ChatOpenAI(model_name='gpt-4o')  
                chain = load_qa_chain(llm=llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs_with_sources, question=onedrive_question_input)
                    print(cb)

                # Add inline sources
                st.write(response)
                if relevant_sources:
                    st.write("### Relevant Sources:")
                    for source in relevant_sources:
                        st.markdown(f"- [{source}]({source})")

                st.write("### Expand for Detailed Content:")
                for idx, doc in enumerate(docs_with_sources):
                    source = doc.metadata.get("url", "Unknown source")
                    url = doc.metadata.get("url", "#")
                    content_full = doc.page_content

                    with st.expander(f"Details from {source} (Click to Expand)"):
                        st.write(f"**Source Document**: [{source}]({url})")  
                        st.write(content_full)  

                data = []
                for doc in docs_with_sources:
                    source = doc.metadata.get("source", "Unknown source")
                    url = doc.metadata.get("url", "#")
                    content_preview = (doc.page_content[:200] + '...') if len(
                        doc.page_content) > 100 else doc.page_content
                    data.append([source, url, content_preview])

            elif model_name == "llama3.1":

                print(f"Powerd by LLAMA query={onedrive_question_input}")

                metadata=[]
                data = []
                for doc in docs_with_sources:
                    source = doc.metadata.get("source", "Unknown source")
                    url = doc.metadata.get("url", "#")
                    content_preview = (doc.page_content[:200] + '...') if len(
                        doc.page_content) > 100 else doc.page_content
                    data.append([source, url, content_preview])
                    metadata.append({"source":source, "url":url,"content_preview":content_preview})

                prompt = knowledgebase_prompt_template.format(input_documents=docs_with_sources,metadata=metadata,question=onedrive_question_input)
                result = model_invoker.call_model(model_name,prompt,"system","Employee Knowledge base lookup assistent")

            else:
                st.warning("Selected model is not supported")
def run():
    with st.sidebar:
    # st.title("**APMT Terminals**")
     st.info('''
        ## Ask Questions to LLaMA from OneDrive Files

        **:blue[Some prominent prompts]** 
        - Outline APMT 2030 Strategy
        - What are some key challenges for APMT 2030 strategy?
        - Show proxy names used in CDH services?
        - Outline the steps to deploy a microservice in CDH TPC?
        ''')

    # CSS to position the "POWERED BY LLM" text at the bottom-left corner
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

    # Injecting custom CSS for table styling
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

    main()
