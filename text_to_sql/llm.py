from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import os
from config import Config
import requests
import json

config = Config()

llama_config = config.get_api_llama_config()

class LLaMAAPI:
    def __init__(self, model_url):
        self.model_url = model_url

    def call_llama_api(self, prompt):
        llama_api_url = self.model_url

        data = {
            "model": "llama3.1",
            "prompt": prompt,
            "max_tokens": 2000,
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
        
sql_prompt_template = PromptTemplate(
    input_variables=["input_question", "context_explanation", "examples"],
    template="""
    You are an SQL expert. Based on the examples and context provided below, generate a valid SQL query using the specified tables
    and columns.

    Examples:
    {examples}

    Now, generate the SQL query for the following input:

    Question: {input_question}

    SQL Query:

    give me generated sql as an output and no other text.
    """
)

def query_openai_model(sql_query, query_result):
    openai_api_url = llama_config['openai_endpoint']
    openai_api_key =config.get_api_keys()["openai_api_key"]
    prompt = layman_prompt.format(sql_query=sql_query, query_result=query_result.to_string)
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
        "prompt": prompt,
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    response = requests.post(openai_api_url, headers=headers, json=data)
    st.write(response.json())
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "")
    else:
        return f"Error querying OpenAI API: {response.status_code}"
    
def generate_layman_explanation(sql_query, query_result):
    try:
        response = query_openai_model(sql_query,query_result)
        explanation = response.get('text', '').strip() if isinstance(response, dict) else response.strip()
        return explanation if explanation else "No explanation could be generated."
    except Exception as e:
        st.error(f"Error generating layman explanation: {e}")
        return None

def init_llm_chain():
    openai_api_key =config.get_api_keys()["openai_api_key"]
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
    llm_chain = LLMChain(llm=llm, prompt=sql_prompt_template)
    return llm_chain


layman_prompt = PromptTemplate(
    input_variables=["sql_query", "query_result"],
    template="""
    Based on the following SQL query and its results, explain in layman terms what the query is doing and what the result means
    for business users.

    SQL Query: {sql_query}
    Query Result: {query_result}

    Explanation for business users:
    """
)

def init_layman_chain(raw_response,query_result):
    prompt = layman_prompt.format(sql_query=raw_response, query_result=query_result.to_string)
    openai_api_key =config.get_api_keys()["openai_api_key"]
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
    layman_chain = LLMChain(llm=llm, prompt=prompt)
    return layman_chain

def init_business_friendly_chain(raw_response,query_result):
    model_url = llama_config['api_endpoint']
    llm = LLaMAAPI(model_url)
    prompt = layman_prompt.format(sql_query=raw_response, query_result=query_result.to_string)

    explanation = llm.call_llama_api(prompt)
    return explanation