import os
import yaml
import re

class Config:
    def __init__(self, config_file="config.yml"):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        return self.replace_placeholders(config)

    def replace_placeholders(self, config):
        if isinstance(config, dict):
            return {key: self.replace_placeholders(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self.replace_placeholders(item) for item in config]
        elif isinstance(config, str):
            return re.sub(r"\$\{(\w+)\}", lambda match: os.getenv(match.group(1), match.group(0)), config)
        else:
            return config

    def get_azure_credentials(self):
        return self.config_data['azure']

    def get_api_keys(self):
        return self.config_data['api_keys']

    def get_available_models(self):
        return self.config_data['models']['available_models']

    def get_database_config(self):
        return self.config_data['database']

    def get_api_integration_auth_token_config(self):
        return self.config_data['api_integration']['token']
    
    def get_api_llama_config(self):
        return self.config_data['llama']
    
    def get_confluence_config(self):
        return self.config_data['confluence']