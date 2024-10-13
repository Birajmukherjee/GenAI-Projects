import os
import threading
import time
import PyPDF2
from docx import Document

indexed_files = {}

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            for page in range(reader.getNumPages()):
                text += reader.getPage(page).extractText()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            text = file.read()
    return text

def index_folder(folder_path):
    global indexed_files
    new_files_indexed = False
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            last_modified_time = os.path.getmtime(file_path)

            if (file_path not in indexed_files) or (indexed_files[file_path] != last_modified_time):
                text = extract_text_from_file(file_path)
                indexed_files[file_path] = {"content": text, "last_modified": last_modified_time}
                print(f"Indexed {file_path}")
                new_files_indexed = True

    return new_files_indexed

def scan_and_index_folder(folder_path, interval=10):
    while True:
        index_folder(folder_path)
        time.sleep(interval)

def start_indexing_thread(folder_path):
    indexing_thread = threading.Thread(target=scan_and_index_folder, args=(folder_path,), daemon=True)
    indexing_thread.start()
    print("Started background indexing thread.")

def search_documents(query):
    results = []
    for file_path, data in indexed_files.items():
        if query.lower() in data["content"].lower():
            results.append(f"Found in {file_path}")
    return results if results else "No relevant documents found."
