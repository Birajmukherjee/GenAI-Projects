
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import json
from langchain.callbacks.base import BaseCallbackHandler

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.embeddings import HuggingFaceEmbeddings    
from langchain.vectorstores import Chroma

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import os

# Display the logo in the sidebar

# Path to the uploaded logo image

# Display the logo

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

persist_directory="./chroma_db_confluence"

# Sidebar contents
with st.sidebar:
    #st.title("**APMT Terminals**")
    st.markdown('''
    ## About:
    **Revolutionizing Legal Workflows with Intelligent Document Analysis**
    
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

    #add_vertical_space(5)
    #st.write('Created by Biraj Mukherjee')

load_dotenv()

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
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


def load_confluence_data(chromaVectorStore):
    
    loader = ConfluenceLoader(
        url="",
        username="",
        api_key=""
    )

    # Load documents from a specific space, including attachments, with a limit of 50 documents
    documents = loader.load(
        space_key="ACDH",
        include_attachments=True,
        limit=50,
        max_pages=50
    )

    print(len(documents))
    for doc in documents:
        
        if doc.page_content is not None and len(doc.page_content.strip()) > 0:
            doc_id = doc.metadata['id']
            print(f"Document Id {doc_id}")
            print(doc)

            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=500,
            #     chunk_overlap=20,
            #     length_function=len,
            #     add_start_index=True)

            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
            chunks = text_splitter.split_text(doc.page_content)


            #metadatas = [{"source": file, "url": f"file://{os.path.abspath(file_path)}"} for _ in chunks]

            # Store metadata separately in a dictionary
            #metadata_store[file] = metadatas
        
            #print(len(chunks))

            # Embed and store the chunks in Chroma DB
            i = 0
            for chunk in chunks:
                print(chunk)
                chromaVectorStore.add_texts(
                    texts=[chunk],
                    ids=[f"{doc_id}_{i}"],
                    metadatas=
                        [
                            doc.metadata
                        ],
                )
                i=i+1

def main():
    st.header("AI-Powered Employee Knowledgebase")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    llm = LlamaCpp(
        model_path="/Users/param.mk/apmt/workspace/ai/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,
        callback_manager=callback_manager, 
        verbose=True, # Verbose is required to pass to the callback manager
    )

    chromaVectorStore=Chroma(collection_name="apmt_confluence_store",
        embedding_function=embeddings,
        persist_directory="./chroma_db_confluence"
    )

    load_confluence_flag=False
    if load_confluence_flag == True:
        load_confluence_data(chromaVectorStore)

    # upload a PDF file
    #pdf = st.file_uploader("Upload your PDF", type='pdf')
    # docs_as_str = [doc.page_content for doc in docs]
    # embedded_docs = embeddings.embed_documents(docs_as_str)
    # chromaVectorStore.add_documents(embedded_docs)

    # Accept user questions/query
    query = st.text_input("Ask questions about your confluence pages and contents:")
     # st.write(query)

    if query:
        #docs = VectorStore.similarity_search(query=query, k=10)

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=chromaVectorStore.as_retriever(),
            verbose=True,
            return_source_documents=True
        )
    
        #response = qa(query)

        # Write answer and sources
        retrieval_streamer_cb = PrintRetrievalHandler(st.container())
        answer = st.empty()
        stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
        response = qa(query, callbacks=[retrieval_streamer_cb, stream_handler])
        answer.info('`Answer:`\n\n' + response['result'])
        st.info('`Sources:`\n\n' + str(response['source_documents']))

if __name__ == '__main__':
    main()