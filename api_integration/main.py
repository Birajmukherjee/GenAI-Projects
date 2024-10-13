import os
import requests
import pandas as pd
from api_integration import api_documentation
from core.auth.auth import APIOAuthClient
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import APIChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.base import LLM
from typing import List
import json
import re
from urllib.parse import urlencode
from config import Config
from langchain.prompts import PromptTemplate

config = Config()
load_dotenv()

llama_config = config.get_api_llama_config()

error_prompt_template = PromptTemplate(
    input_variables=["error_message"],
    template="""
        Explain the following API error message in a way that is easy for users to understand. 
        Focus on the reasons specific to a 404 status code, which means the resource was not found. 
        Avoid mentioning authentication issues like API keys, which usually result in a 401 or 403 error.
        Provide possible reasons for the 404 error and steps users can take to resolve it.

        Error message: {error_message}

        Answer:
    """
)
success_prompt_template = PromptTemplate(
    input_variables=["api_response"],
    template="""
       "prompt": "Using the following API response, 
       create a detailed and well-organized journey summary for the container.
         The summary should focus on the most recent events first (latest data first). 
         Important details such as **Container ID**, **Carrier**, **Origin**, **Destination**, and **Event Dates** should be highlighted using bold formatting.
         Events like *Arrival*, *Discharge*, *Final Delivery*, and *Departure* should be italicized.
        The response should be easy to read, avoiding any mention of the prompt or example generation, and should directly present the results based on the data.

    API Response: {api_response}\n\n\
    Answer:"
    """
)


def enrich_error_message(error):
    llama_api_url = llama_config['api_endpoint']
    error_message_prompt = error_prompt_template.format(error_message=error)
    data = {
        "model": "llama3.1",
        "prompt": error_message_prompt,
        "max_tokens": 2000,
        "temperature": 0.3,  
        "top_p": 0.9,
    }
    response_placeholder = st.empty()

    response = requests.post(llama_api_url, json=data, stream=True)
    if response.status_code == 200:
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    chunk_data = chunk.decode("utf-8")
                    chunk_json = json.loads(chunk_data)
                    token = chunk_json.get("response", "")
                    full_response += token
                    response_placeholder.info(full_response)
                except json.JSONDecodeError:
                    st.error("Failed to decode JSON from LLaMA API")
                    break
    return full_response

def enrich_success_message(response):
    llama_api_url = llama_config['api_endpoint']
    success_message_prompt = success_prompt_template.format(api_response=response)
    data = {
        "model": "llama3.1",
        "prompt": success_message_prompt,
        "max_tokens": 2000,
        "temperature": 0.3,  
        "top_p": 0.9,
    }
    response_placeholder = st.empty()

    response = requests.post(llama_api_url, json=data, stream=True)
    if response.status_code == 200:
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    chunk_data = chunk.decode("utf-8")
                    chunk_json = json.loads(chunk_data)
                    token = chunk_json.get("response", "")
                    full_response += token
                    response_placeholder.info(full_response)
                except json.JSONDecodeError:
                    st.error("Failed to decode JSON from LLaMA API")
                    break
    return full_response


class LLaMAAPI(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        llama_api_url = llama_config['api_endpoint']

        new_prompt = f"Provide only the final API URL for this query: {prompt}. No explanation needed, just the URL."

        data = {
            "model": "llama3.1",
            "prompt": new_prompt,
            "max_tokens": 200,
            "temperature": 0.3,  
            "top_p": 0.9,
        }
        response_placeholder = st.empty()

        response = requests.post(llama_api_url, json=data, stream=True)

        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        chunk_data = chunk.decode("utf-8")
                        chunk_json = json.loads(chunk_data)
                        token = chunk_json.get("response", "")
                        full_response += token
                    except json.JSONDecodeError:
                        st.error("Failed to decode JSON from LLaMA API")
                        break
            return full_response
        else:
            return f"Error querying LLaMA API: {response.status_code}"

    @property
    def _identifying_params(self) -> dict:
        return {"name_of_llm": "LLaMAAPI"}

    @property
    def _llm_type(self) -> str:
        return "custom_llama_api"

def extract_api_url(text: str) -> str:
    url_match = re.search(r"https?://[^\s]+", text)
    if url_match:
        return url_match.group(0)
    else:
        return None

def run():
    #st.header("AI-Powered API Query with LLaMA and API Calls")
    if "question" not in st.session_state:
        st.session_state["question"] = None

    if "query" not in st.session_state:
        st.session_state["query"] = None

    d = {
        'API Endpoint': ['/container-event-history', '/import-availability', '/vessel-visits', '/all-vessel-schedules'],
        'Description': [
            'This API returns time-stamped description of events registered against container ID(s).',
            'This API tells whether a container is available for import or not at a specific terminal location.',
            'This API returns Vessel Visits by voyage numbers.',
            'This API provides the list of all vessels and their schedule for a given terminal and within a specific time range (-6 days to +14 days).'
        ]
    }
    df = pd.DataFrame.from_dict(d)
    with st.sidebar:
        st.info('''
            ## Intelligent T&T Visibility:
            **The Intelligent Concession Analyst enables us to analyze legal clauses quickly and make better decisions when dealing with contracts.**   

            **:blue[Some prominent prompts]** 
            - Import availability of container FCIU7465680 at Gothenburg
            - Provide vessel visit information for vessel Mate at terminal Los Angeles
            - List of  container events happened on container MSDU6341431 at Los Angeles             
        ''')
        st.sidebar.table(df)


    #st.info("`I am an AI that can answer questions by analyzing and converting questions into API requests and generating responses.`")

    llm = LLaMAAPI()

    api_question_input = st.text_input("What you love to know (API):")

    if api_question_input:
        st.write(f"Searching for: {api_question_input}")

        valid_domains = [
            "https://api.apmterminals.com"
        ]
        endpoint_details=config.get_api_integration_auth_token_config()
        oauth_client = APIOAuthClient(endpoint_details['client_id'], endpoint_details['secret'], endpoint_details['url'])
        access_token = oauth_client.get_access_token()

        headers = {"Authorization": f"Bearer {access_token}"}

        qa_chain = APIChain.from_llm_and_api_docs(llm, api_documentation.api_docs(),headers=headers, limit_to_domains=valid_domains, verbose=True)

        answer = st.empty()
        result = qa_chain.run(api_question_input)
        print(result)
        api_url = extract_api_url(result)

        if api_url:
            st.write(f"Generated API URL: {api_url}")
            try:
                if access_token:
                    response = requests.get(api_url, headers=headers)
                    if response.status_code == 200:
                        print(response.json())
                        enrich_success_message(f"{response.json()}")
                    else:
                        enrich_error_message(response)
                else:
                    print('Failed to get access token')
                
            except Exception as e:
                st.error(f"Error making API request: {str(e)}")
        else:
            st.error("No valid API URL found in the generated response.")

if __name__ == "__main__":
    run()
