from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from text_to_sql.entity_data import Entity_joins
import os
from config import Config

config = Config()


def init_faiss_store():
    openai_api_key = config.get_api_keys()["openai_api_key"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    texts = [item['query'] for item in Entity_joins]
    metadatas = [{"sql_query": item['sql']} for item in Entity_joins]

    faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return faiss_index
