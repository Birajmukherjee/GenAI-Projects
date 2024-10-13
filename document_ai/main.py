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
from config import Config
from core.llm.llm_invoker import ModelInvoker
from langchain.prompts import PromptTemplate

config = Config()
llama_config = config.get_api_llama_config()
model_invoker = ModelInvoker()

# Display the logo in the sidebar
# st.sidebar.image("apm_terminal.png", width=200)

# Path to the uploaded logo image
# logo_path = "genai.jpeg"

# Display the logo
# st.image(logo_path, width = 150)

# Sidebar contents

load_dotenv()

documentbase_prompt_template = PromptTemplate(
    input_variables=["input_documents", "metadata", "question"],
    template="""
    I am an employee looking to get the constructive information of the different clauses from the legal documents along with links & references which describes about that question given below.
    Based on the content provided below, generate a structured and relevant response specific to clauses of each Prompt. Organize the information into categories such as "Clauses", "Articles", "differecne between Clauses", "list all Clause family with detailed explanations", etc., as appropriate. Only include relevant information from the content.

    Content:
    {input_documents}

    Metadata: {metadata}

    Now, generate an optimal response for the following question or statement:

    Question: {question}

    Please include the **document links** as part of the source in the following format:

    you are an document AI chat bot, based on the "Prominent Prompts" provided below, you have to paraphase and articulate Clauses and its related families and detailed explanation from the provided document.
    PROMINENT_PROMPTS = [
    "Tariff terms",
    "Agreement Term",
    "Force Majeure",
    "Handback & Surrendering",
    "Insurance & Liabilities",
    "Maintenance Obligations",
    "Payment Terms",
    "Performance Requirements",
    "Term Extension",
    "Financial Equilibrium",
    "Ownership Requirements",
    "Infrastructure Obligations",
    "Equipment Obligations",
    "Labour Nationalization Requirement"
]

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
    # Get the absolute path
    abs_path = os.path.abspath(file_path)

    # On Windows, replace backslashes with forward slashes
    if os.name == 'nt':
        abs_path = abs_path.replace('\\', '/')

    # Create the URL using urllib
    file_url = urllib.parse.urljoin('file:', urllib.parse.quote(abs_path))

    return file_url


def main():
    if "question" not in st.session_state:
        st.session_state["question"] = None

    if "query" not in st.session_state:
        st.session_state["query"] = None

    global applicationDatasourceDocsMemory
    #st.header("Intelligent Concession Management System")

    # Directory containing PDF and JSON files
    directory_path = "./document_ai_source_documents"

    # Check if the directory exists
    if not os.path.exists(directory_path):
        st.error(f"The directory '{directory_path}' does not exist.")
        return

    # List all PDF and JSON files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith(('.pdf', '.json'))]

    if not files:
        st.warning("No PDF or JSON files found in the directory.")
        return

    vector_stores = {}
    metadata_store = {}

    # Ensure the global variable is initialized
    applicationDatasourceDocsMemory = None

    # Process each file in the directory
    for file in files:
        file_path = os.path.join(directory_path, file)

        # Ensure the file exists before processing
        if not os.path.exists(file_path):
            st.warning(f"File {file} not found. Skipping.")
            continue

        # Extract text depending on the file type
        if file.endswith('.pdf'):
            text = process_pdf(file_path)
        elif file.endswith('.json'):
            text = process_json(file_path)

        # Split the extracted text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create metadata with the source document name and hyperlink
        metadatas = [{"source": file, "url": f"file://{os.path.abspath(file_path)}"} for _ in chunks]

        # Store metadata separately in a dictionary
        metadata_store[file] = metadatas

        # Define a unique store name based on the file name
        store_name = file[:-4]  # Remove the extension
        print(f"Store name : {store_name}")
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings() 
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            #datasource_retriever = VectorStore.as_retriever(search_type="mmr")
            #applicationDatasourceDocsMemory = VectorStoreRetrieverMemory(retriever=datasource_retriever)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Create retriever and memory store even if the VectorStore is loaded from pickle
        datasource_retriever = VectorStore.as_retriever(search_type="mmr")
        applicationDatasourceDocsMemory = VectorStoreRetrieverMemory(retriever=datasource_retriever)

        # Store the vector store for each document
        vector_stores[file] = VectorStore

    # Save the metadata store to a file
    with open("metadata_store.pkl", "wb") as f:
        pickle.dump(metadata_store, f)

    # Accept user questions/query
    document_ai_question_input = st.text_input("What you love to know (Document Intelligence):")

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
                source = doc.metadata.get("source", "Unknown source")
                with self.container.expander(f"Results from {source} (Click to Expand)"):
                    # Display document content and metadata
                    self.container.write(f"**Source Document:** {source}")
                    self.container.write(doc.page_content)  # Show the full context inside the expande

    if document_ai_question_input:
        # Load the metadata store from the file
        with open("metadata_store.pkl", "rb") as f:
            metadata_store = pickle.load(f)

        # Search across all vector stores
        docs_with_sources = []
        for file, vector_store in vector_stores.items():
            results = vector_store.similarity_search(query=document_ai_question_input, k=3)
            for result in results:
                chunk_idx = result.metadata.get("chunk_index", 0)
                # Attach the metadata from the external store
                result.metadata.update(metadata_store[file][chunk_idx])
                docs_with_sources.append(result)

        model_name=st.session_state['model_name']

        if model_name == "OpenAI":
            print(f"Powerd by OpenAI GPT-4o query={document_ai_question_input}")
            llm = ChatOpenAI(model_name='gpt-4o')  # Use the correct model name
            chain = load_qa_chain(llm=llm, chain_type="stuff")


            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs_with_sources, question=document_ai_question_input)
                print(cb)

            st.write(response)
            st.write("### Sources:")

            st.write("### Expand for Detailed Content:")
            for idx, doc in enumerate(docs_with_sources):
                source = doc.metadata.get("source", "Unknown source")
                url = doc.metadata.get("url", "#")
                content_full = doc.page_content

                with st.expander(f"Details from {source} (Click to Expand)"):
                    st.write(f"**Source Document**: [{source}]({url})")  # Show source with hyperlink
                    st.write(content_full)  # Show the full document content inside the expander
            
            metadata= []
            data = []
            for doc in docs_with_sources:
                source = doc.metadata.get("source", "Unknown source")
                url = doc.metadata.get("url", "#")
                content_preview = (doc.page_content[:200] + '...') if len(
                    doc.page_content) > 100 else doc.page_content
                data.append([source, url, content_preview])

        elif model_name == "llama3.1":

            print(f"Powerd by LLAMA query={document_ai_question_input}")

            # response = chain.run(input_documents=docs_with_sources, question=document_ai_question_input)

            # st.write(response)
            # st.write("### Sources:")

            # st.write("### Expand for Detailed Content:")
            # for idx, doc in enumerate(docs_with_sources):
            #     source = doc.metadata.get("source", "Unknown source")
            #     url = doc.metadata.get("url", "#")
            #     content_full = doc.page_content

            #     with st.expander(f"Details from {source} (Click to Expand)"):
            #         st.write(f"**Source Document**: [{source}]({url})")  # Show source with hyperlink
            #         st.write(content_full)  # Show the full document content inside the expander

            metadata=[]
            data = []
            for doc in docs_with_sources:
                source = doc.metadata.get("source", "Unknown source")
                url = doc.metadata.get("url", "#")
                content_preview = (doc.page_content[:200] + '...') if len(
                    doc.page_content) > 100 else doc.page_content
                data.append([source, url, content_preview])
                metadata.append({"source":source, "url":url,"content_preview":content_preview})

            prompt = documentbase_prompt_template.format(input_documents=docs_with_sources,metadata=metadata,question=document_ai_question_input)
            result = model_invoker.call_model(model_name,prompt,"system","AI Assistent")

        else:
            st.warning("Selected model is not supported")    

def run():

    with st.sidebar:
    # st.title("**APMT Terminals**")
     st.info('''
        ## Intelligent Concession Analyst:
        **The Intelligent Concession Analyst enables us to analyze legal clauses quickly and make better decisions when dealing with contracts.**  
                
        **:blue[Some prominent prompts]** 
        - Please provide a summary of the key terms and conditions outlined in the terminal port concession document.
        - Analyze the financial implications of the concession agreement, including revenue-sharing models and cost structures.
        - List the compliance requirements that the concessionaire must adhere to as per the concession agreement.
        - What performance metrics are specified in the concession agreement to evaluate the concessionaire's effectiveness?
        - Outline the legal obligations of both parties as stated in the terminal port concession document.
        - Describe the legal provisions included in the concession agreement.
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