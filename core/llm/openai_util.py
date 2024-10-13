# openai_util.py

import requests
import json
import streamlit as st

class OpenAIAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def call_openai_api(self, prompt,role="system",content="AI Assistent"):
        openai_api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4", 
            "messages": [
                {"role": role, "content": content},  
                {"role": "user", "content": prompt}  
            ],
            "max_tokens": 5000,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": True  
        }

        response_placeholder = st.empty()  
        full_response = ""

        response = requests.post(openai_api_url, headers=headers, json=data, stream=True)

        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    if chunk.strip() == "data: [DONE]":
                        break

                    try:
                        chunk_data = chunk.decode("utf-8").replace("data: ", "")
                        chunk_json = json.loads(chunk_data)

                        token = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        full_response += token

                        response_placeholder.write(f"Answer: {full_response}")
                    except json.JSONDecodeError:
                        continue 
            return full_response
        else:
            st.error(f"Error querying OpenAI API: {response.status_code}, {response.text}")
            return None
