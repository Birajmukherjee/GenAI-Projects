# llama_util.py

import requests
import streamlit as st
import json

class LLaMAAPI:
    def __init__(self, model_url):
        self.model_url = model_url

    def call_llama_api(self, prompt,role,content):
        data = {
            "model": "llama3.1",
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.2,
            "best_of": 3
        }
        response_placeholder = st.empty()
        response = requests.post(self.model_url, json=data, stream=True)

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
