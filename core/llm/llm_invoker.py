
from config import Config
from core.llm.llama_util import LLaMAAPI
from core.llm.openai_util import OpenAIAPI

class ModelInvoker:
    def __init__(self):
        self.config = Config()
        self.llama_models = {
            "llama3.1": LLaMAAPI(model_url=self.config.get_api_llama_config()['api_endpoint']),
            "llama3.1:405b": LLaMAAPI(model_url=self.config.get_api_llama_config()['api_endpoint_405b'])
        }
        self.openai_model = OpenAIAPI(api_key=self.config.get_api_keys()["openai_api_key"])

    def call_model(self, model_name, prompt,role,content):
        if model_name in self.llama_models:
            print("Powered by LLAMA")
            return self.llama_models[model_name].call_llama_api(prompt,role,content)
        elif model_name == "OPENAI":
            print("Powered by OpenAI")
            return self.openai_model.call_openai_api(prompt,role,content)
        else:
            raise ValueError(f"Model '{model_name}' not supported.")
