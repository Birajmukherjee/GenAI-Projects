import re
import os
import json
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
from langchain.chains.question_answering import load_qa_chain, chain
from langchain.callbacks import get_openai_callback
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import ConfluenceLoader
from config import Config
from core.llm.llm_invoker import ModelInvoker
from langchain.prompts import PromptTemplate

config = Config()
llama_config = config.get_api_llama_config()
model_invoker = ModelInvoker()

confluence_local_store = "confluence_local_store"

    # add_vertical_space(5)
    # st.write('Created by Biraj Mukherjee')

# Load environment variables (e.g., for the OpenAI API key)
load_dotenv()
knowledgebase_prompt_template = PromptTemplate(
    input_variables=["input_documents", "metadata", "question"],
    template="""
    I am an employee looking to get the constructive information from the company documents along with links & references which describes about that question given below.
    Based on the content provided below, generate a structured and relevant response. Organize the information into categories such as "Performance Testing", "Database Analysis", "Data Integration", "Code Generation", etc., as appropriate. Only include relevant information from the content.

    Content:
    {input_documents}

    Metadata: {metadata}

    Now, generate an optimal response for the following question or statement:

    Question: {question}

    Please include the **document links** as part of the source in the following format:

    Performance Testing:
    - [Activity description] (Source: [Document Link])

    Ensure that you extract and display the links from the metadata, if available. If no link is available, just display "(Source: No link available)".
    """
)



def process_pdf(pdf_path):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def process_json(json_path):
    """Extract text from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Flatten the JSON structure to a string (customize based on your JSON structure)
    text = json.dumps(data, indent=2)
    return text


def create_file_url(file_path):
    """Create a properly formatted file URL."""
    abs_path = os.path.abspath(file_path)

    if os.name == 'nt':
        abs_path = abs_path.replace('\\', '/')

    file_url = urllib.parse.urljoin('file:', urllib.parse.quote(abs_path))

    return file_url


def convert_page_name_to_file_name(page_name):
    file_name = page_name.replace(" ", "_")
    file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)
    return file_name



def main():
    if "question" not in st.session_state:
        st.session_state["question"] = None

    if "query" not in st.session_state:
        st.session_state["query"] = None
        
    config = Config()

    confluence_config = config.get_confluence_config()
    global applicationDatasourceDocsMemory
    #st.header("Intelligent Concession Management System")
    vector_stores = {}
    metadata_store = {}
    if "apmt_confluence_store" not in st.session_state:
        loader = ConfluenceLoader(
                url=confluence_config['url'],
                username=confluence_config['username'],
                api_key=confluence_config['auth_token']
            )

        documents = loader.load(
            space_key="ACDH",
            include_attachments=True,
            limit=10,
            max_pages=10
        )
        try:
            for doc in documents:
                if doc.page_content is not None and len(doc.page_content.strip()) > 0:
                    doc_id = doc.metadata['id']
                    print(f"Document Id {doc_id}")

                applicationDatasourceDocsMemory = None

                # Split the extracted text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=doc.page_content)
                if not chunks:
                    print(f"No text chunks were generated for document: {doc.metadata.get('title')}")
                    continue

                if not doc.page_content or len(doc.page_content.strip()) == 0:
                    print(f"Document {doc.metadata.get('title')} has no content.")
                    continue

                title =doc.metadata.get('title')
                source = doc.metadata.get('source')
                when = doc.metadata.get('when')
                store_name = convert_page_name_to_file_name(title)

                # Create metadata with the source document name and hyperlink
                metadatas = [{
                            "title": title,
                            "source": source,
                            "when": when} for _ in chunks]

                metadata_store[store_name] = metadatas

                print(f"Store name : {store_name}")
                if os.path.exists(f"{confluence_local_store}/{store_name}.pkl"):
                    with open(f"{confluence_local_store}/{store_name}.pkl", "rb") as f:
                        vector_stores[store_name] = pickle.load(f)
                else:
                    embeddings = OpenAIEmbeddings() 
                    vector_stores[store_name] = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{confluence_local_store}/{store_name}.pkl", "wb") as f:
                        pickle.dump(vector_stores[store_name], f)

                st.session_state["apmt_confluence_store"] = "SUCCESS"
                st.session_state["vector_stores"] = vector_stores
                st.session_state["metadata_store"] = metadata_store
                with open(f"{confluence_local_store}/metadata_store.pkl", "wb") as f:
                    pickle.dump(metadata_store, f)

        except IndexError as e:
            print(f"Error creating vector store for {store_name}: {e}")
            st.error(f"Failed to create vector store for {store_name}. Please check the embeddings.")
            return
    else:
        vector_stores = st.session_state["vector_stores"]
        metadata_store = st.session_state["metadata_store"]

    confluence_kb_question_input = st.text_input("What you love to know (Confluence):")

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
            # Iterate over the retrieved documents and create expanders for each
            for idx, doc in enumerate(documents):
                source,title,when = doc.metadata.get("source", "title","when")
                self.container.write(f"**Source Document:** {title} source{source} {when}")

    if confluence_kb_question_input:
        # Load the metadata store from the file
        with open(f"{confluence_local_store}/metadata_store.pkl", "rb") as f:
            metadata_store = pickle.load(f)

        # Search across all vector stores
        docs_with_sources = []
        for title, vector_store in vector_stores.items():
            results = vector_store.similarity_search(query=confluence_kb_question_input, k=3)
            for result in results:
                chunk_idx = result.metadata.get("chunk_index", 0)
                result.metadata.update(metadata_store[convert_page_name_to_file_name(title)][chunk_idx])
                docs_with_sources.append(result)
    
        model_name=st.session_state['model_name']

        if model_name == "OpenAI":
            print(f"Powerd by OpenAI GPT-4o query={confluence_kb_question_input}")
            llm = ChatOpenAI(model_name='gpt-4o')  # Use the correct model name
            chain = load_qa_chain(llm=llm, chain_type="stuff")


            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs_with_sources, question=confluence_kb_question_input)
                print(cb)

            st.write(response)
            st.write("### Sources:")
            st.write("### Expand for Detailed content:")
            for idx, doc in enumerate(docs_with_sources):
                source = doc.metadata.get("source")
                title = doc.metadata.get("title")
                when = doc.metadata.get("when")
                content_full = doc.page_content

                with st.expander(f"Details from {source}  (Click to Expand)"):
                    st.write(f"**Source Document**: [{source}]")  # Show source with hyperlink
                    st.write(f"**Title {title} - when {when}")
                    st.write(content_full)  # Show the full document content inside the expander

            metadata=[]
            data = []
            for doc in docs_with_sources:
                source = doc.metadata.get("source", "Unknown source")
                url = doc.metadata.get("url", "#")
                content_preview = (doc.page_content[:200] + '...') if len(
                    doc.page_content) > 100 else doc.page_content
                data.append([source, url, content_preview])
                metadata.append({"source":source, "url":url,"content_preview":content_preview})

        elif model_name == "llama3.1":

            print(f"Powerd by LLAMA query={confluence_kb_question_input}")

            metadata=[]
            data = []
            for doc in docs_with_sources:
                source = doc.metadata.get("source", "Unknown source")
                url = doc.metadata.get("url", "#")
                content_preview = (doc.page_content[:200] + '...') if len(
                    doc.page_content) > 100 else doc.page_content
                data.append([source, url, content_preview])
                metadata.append({"source":source, "url":url,"content_preview":content_preview})

            prompt = knowledgebase_prompt_template.format(input_documents=docs_with_sources,metadata=metadata,question=confluence_kb_question_input)
            result = model_invoker.call_model(model_name,prompt,"system","AI Assistent")

        else:
            st.warning("Selected model is not supported")


def run():
    # Sidebar contents
    with st.sidebar:
        # st.title("**APMT Terminals**")
        st.info('''
        ## Intelligent Employee Knowledge Base ChatBot:
        **The Knowledge Base Bot ensures employees can access the information they need instantly, improving productivity and consistency**

        **:blue[Some prominent prompts]** 
        - Explain the key components of Central data hub 2.0 system
        - What are the activities performed by APMT hybrid cloud?
        - Outline the steps to deploy microservice in Hybrid cloud        
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