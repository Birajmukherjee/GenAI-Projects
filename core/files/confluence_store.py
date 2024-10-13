import streamlit as st
import os
import pickle
import re
from langchain.embeddings import HuggingFaceEmbeddings    
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from config import Config

config = Config()

confluence_config = config.get_confluence_config()

confluence_local_store = "confluence_local_store"

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

@st.cache_data
def load_stores_from_disk():
    vector_stores = {}
    metadata_store = {}

    if os.path.exists(f"{confluence_local_store}/vector_stores.pkl") and os.path.exists(f"{confluence_local_store}/metadata_store.pkl"):
        with open(f"{confluence_local_store}/vector_stores.pkl", "rb") as f:
            vector_stores = pickle.load(f)
        with open(f"{confluence_local_store}/metadata_store.pkl", "rb") as f:
            metadata_store = pickle.load(f)
        print("Loaded vector stores and metadata from disk.")
        st.session_state["apmt_confluence_store"] = True
    else:
        print("No stores found on disk. loading data...")
    
    return vector_stores, metadata_store

@st.cache_data
def save_stores_to_disk(_vector_stores, _metadata_store):
    with open(f"{confluence_local_store}/vector_stores.pkl", "wb") as f:
        pickle.dump(_vector_stores, f)
    with open(f"{confluence_local_store}/metadata_store.pkl", "wb") as f:
        pickle.dump(_metadata_store, f)
    print("Vector stores and metadata saved to disk.")

class CustomVectorStore:

    def __init__(self):
        self.vector_stores = {}
        self.metadata_store = {}
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name,
                                                encode_kwargs=self.encode_kwargs,
                                                model_kwargs=self.model_kwargs)

    def convert_page_name_to_file_name(self, page_name):
        file_name = page_name.replace(" ", "_")
        file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)
        return file_name
    
    def initialize_stores(self):
        ensure_directory_exists(confluence_local_store)
        if 'apmt_confluence_store' not in st.session_state:
            st.session_state["vector_stores"], st.session_state["metadata_store"] = load_stores_from_disk()
            st.session_state["apmt_confluence_store"] = True
        else:
            self.vector_stores = st.session_state["vector_stores"]
            self.metadata_store = st.session_state["metadata_store"]

    def update_and_save_stores(self, new_data):
        self.vector_stores.update(new_data['vector_stores'])
        self.metadata_store.update(new_data['metadata_store'])
        save_stores_to_disk(self.vector_stores,self.metadata_store)

    def get_embeddings(self):
        return self.embeddings

    def get_vector_store(self):
        return self.vector_stores

    def get_metadata_store(self):
        return self.metadata_store

    def load_documents_in_batches(self, loader, batch_size, max_pages):
            documents = loader.load(
                space_key="ACDH",
                include_attachments=True,
                limit=batch_size,
                max_pages=max_pages
            )
            return documents

    def load_confluence_data(self):
        try:
            loader = ConfluenceLoader(
                url=confluence_config['url'],
                username=confluence_config['username'],
                api_key=confluence_config['auth_token']
            )


            documents = self.load_documents_in_batches(loader, 10, 10)
            vector_stores = {}
            metadata_store = {}
            for doc in documents:
                if not doc.page_content or len(doc.page_content.strip()) == 0:
                    continue

                title = doc.metadata.get('title')
                store_name = self.convert_page_name_to_file_name(title)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=doc.page_content)
                metadatas = [{
                    "title": title,
                    "source": doc.metadata.get('source'),
                    "when": doc.metadata.get('when')
                } for _ in chunks]
                
                metadata_store[store_name] = metadatas

                if os.path.exists(f"{store_name}.pkl"):
                    with open(f"{confluence_local_store}/{store_name}.pkl", "rb") as f:
                        vector_stores[store_name] = pickle.load(f)
                else:
                    vector_stores[store_name] = FAISS.from_texts(chunks, embedding=HuggingFaceEmbeddings(model_name=self.model_name,
                                                encode_kwargs=self.encode_kwargs,
                                                model_kwargs=self.model_kwargs))
                    with open(f"{confluence_local_store}/{store_name}.pkl", "wb") as f:
                        pickle.dump(store_name, f)
                    
                    self.vector_stores[store_name] = vector_stores
                    self.metadata_store[store_name] = metadata_store

            save_stores_to_disk(self.vector_stores,self.metadata_store)
            st.session_state["apmt_confluence_store"] = "SUCCESS"

        except Exception as e:
            st.error(f"Error loading Confluence data: {e}")
            raise e

    def load_vector_store(self, is_load):
        """Main entry point to load or initialize the vector store."""
        if is_load and ("apmt_confluence_store" not in st.session_state):
            if not os.path.exists(f"{confluence_local_store}/vector_stores.pkl") and not os.path.exists(f"{confluence_local_store}/metadata_store.pkl"):
                print("Initializing and loading vector store...")
                self.load_confluence_data()
            else:
                load_stores_from_disk()

