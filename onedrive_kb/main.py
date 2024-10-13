import os
import PyPDF2
from docx import Document
from PIL import Image
import pandas as pd
import threading
import time
import requests
import io
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import json
from config import Config
import re

config = Config()
llama_config = config.get_api_llama_config()

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding_dim = 384  
index = faiss.IndexFlatL2(embedding_dim)  

def convert_page_name_to_file_name(page_name):
    file_name = page_name.replace(" ", "_")
    file_name = re.sub(r'[<>:"/\\|?*]', '', file_name)
    return file_name

indexed_files = {}

def extract_text_from_file(file_path, file_content):
    text = ""
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
        text = f"Image file: {file_path} [Image data not extracted]"
    return text

def index_onedrive_folder(api_key, folder_id):
    global indexed_files, index
    headers = {"Authorization": f"Bearer {api_key}"}
    
    response = requests.get(f"https://graph.microsoft.com/v1.0/groups/0b394df8-c35d-4dc0-ad16-4bdf23036baa/drive/items/{folder_id}/children", headers=headers)
    
    if response.status_code == 200:
        files = response.json().get('value', [])
        for file in files:
            file_id = file['id']
            file_name = file['name']
            last_modified = file['lastModifiedDateTime']
            file_link = file['webUrl']  
            
            if file_id not in indexed_files or indexed_files[file_id]['last_modified'] != last_modified:
                file_content_response = requests.get(f"https://graph.microsoft.com/v1.0/groups/0b394df8-c35d-4dc0-ad16-4bdf23036baa/drive/items/{file_id}/content", headers=headers)
                if file_content_response.status_code == 200:
                    file_content = file_content_response.content
                    text = extract_text_from_file(file_name, file_content)
                    
                    embeddings = embed_model.encode([text], convert_to_tensor=False)[0]
                    
                    index.add(embeddings.reshape(1, -1))  
                    
                    indexed_files[file_id] = {
                        "content": text,
                        "last_modified": last_modified,
                        "file_name": file_name,
                        "file_link": file_link
                    }
                    print(f"Indexed {file_name}")
                else:
                    print(f"Failed to download file content: {file_content_response.status_code}")
    else:
        print(f"Failed to list OneDrive folder contents: {response.status_code}")

def periodically_index_folder(api_key, folder_id, interval=300):
    while True:
        print("started one drive indexing")
        index_onedrive_folder(api_key, folder_id)
        print("One drive indexing complete")
        time.sleep(interval)

def start_indexing_thread(api_key, folder_id):
    indexing_thread = threading.Thread(target=periodically_index_folder, args=(api_key, folder_id), daemon=True)
    indexing_thread.start()

def search_indexed_files(query, top_k=1): 
    global index
    query_embedding = embed_model.encode([query], convert_to_tensor=False)[0].reshape(1, -1)
    
    if index.ntotal == 0:
        st.warning("No documents have been indexed yet.")
        return []

    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(indexed_files):
            file_id = list(indexed_files.keys())[idx]  
            results.append({
                "file_name": indexed_files[file_id]["file_name"],
                "content": indexed_files[file_id]["content"][:500], 
                "file_link": indexed_files[file_id]["file_link"] 
            })
    return results

def query_llama_model(prompt):
    llama_api_url = llama_config['api_endpoint']
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    }
    response_placeholder = st.empty()

    response = requests.post(llama_api_url, json=data, stream=True)
    if response.status_code == 200:
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk_data = chunk.decode("utf-8")
                chunk_json = json.loads(chunk_data)
                full_response += chunk_json.get("response", "")
                response_placeholder.info(full_response)
                if chunk_json.get("done", False):
                    break 
        return full_response
    else:
        print(f"Error querying LLaMA model: {response.status_code}, {response.text}")
        return f"Error querying LLaMA model: {response.status_code}"

def run_search_app(api_key, folder_id):
    st.header("Ask Questions to LLaMA from OneDrive Files")
    
    query = st.text_input("Ask AI")
    
    if query:
        results = search_indexed_files(query)
        if results:
            most_relevant_doc = results[0]
            file_name = most_relevant_doc['file_name']
            file_link = most_relevant_doc['file_link']
            content_snippet = most_relevant_doc['content']

            llama_prompt = (
                f"Document: {file_name}\n\n"
                f"Content Snippet: {content_snippet[:500]}\n\n"
                f"OneDrive Link to full document: {file_link}\n\n"
                f"Please include this OneDrive link in the response when answering the following question:\n\n"
                f"Question: {query}"
            )
            llama_response = query_llama_model(llama_prompt)
        else:
            st.write("No matches found in indexed documents.")


def run():
    if st.session_state["api_key"] and st.session_state["folder_id"]:
        run_search_app(st.session_state["api_key"], st.session_state["folder_id"])
    else:
        st.write("Please enter your API key and folder ID to start.")

if __name__ == "__main__":
    run()
