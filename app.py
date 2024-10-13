from onedrive_kb.main import start_indexing_thread
import streamlit as st
import importlib
from config import Config
import msal
import requests
import urllib.parse
from streamlit_msal import Msal

api_key_placeholder = "********"


config = Config()
auth_data = None

def manage_api_key():
    with st.sidebar.expander("OneDrive API Key and Folder Settings", expanded=False):
        api_key = st.text_input("Enter OneDrive API Key", type="password")
        folder_id = st.text_input("Enter OneDrive Folder ID")
        
        if st.button("Start Indexing") and api_key and folder_id:
            st.session_state["api_key"] = api_key
            st.session_state["folder_id"] = folder_id
            start_indexing_thread(api_key, folder_id)

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = config.get_api_keys()['openai_api_key']

def onedrive_kb_integration():
    if st.session_state["api_key"] and st.session_state["folder_id"]:
        start_indexing_thread(st.session_state["api_key"], st.session_state["folder_id"])


st.set_page_config(
    page_title="AI Knowledge base",  
    layout="wide",                
    initial_sidebar_state="expanded"  
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #E74C3C;  
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #C0392B; 
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def load_css():
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_app_bar(user_info=None):
    with open("static/app_bar.html") as f:
        app_bar_html = f.read()
        st.markdown(app_bar_html, unsafe_allow_html=True)

def azure_sign_in():
    client_config = config.get_azure_credentials()
    with st.sidebar:
        auth_data = Msal.initialize_ui(
            client_id=client_config["client_id"],
            authority=client_config["authority"],
            scopes=['User.Read'],
            connecting_label="Connecting",
            disconnected_label="Disconnected",
            sign_in_label="Sign in",
            sign_out_label="Sign out"
        )
    return auth_data

def show_sidebar(user_info):
    st.sidebar.title('AI Knowledge base')

    model = st.sidebar.selectbox("Select Model", ["llama3.1", "OpenAI"])
    st.session_state["user_info"]=user_info
    if user_info:
        selected_option = st.sidebar.radio("Choose an option",
                                            [
                                                "Ask AI", 
                                                "Knowledge Base",
                                                "API Integration",
                                                "Text to SQL",
                                                "OneDrive",
                                                "Feedback",
                                                "Feedback Insights"])

    else:
        selected_option = st.sidebar.radio("Choose an option", ["Ask AI"])
    
    return selected_option, model

def main():
    load_css()
    load_app_bar()
    user_info = azure_sign_in()
    manage_api_key()

    selected_option, model = show_sidebar(user_info)
    st.session_state['model_name']=model
    
    if st.session_state["api_key"] and selected_option=="OneDrive":
        onedrive_kb_integration()

    integration_module = load_integration(selected_option)
    if integration_module:
        integration_module.run()


def load_integration(selected_option):
    option_mapping = {
        "Ask AI": "generic_search.main",
        "Knowledge Base": "confluence_kb.main",
        "API Integration": "api_integration.main",
        "Text to SQL":"text_to_sql.main",
        "OneDrive":"onedrive_kb.main",
        "Feedback":"feedback.main",
        "Feedback Insights":"feedback_insights.main",
    }
    
    if selected_option in option_mapping:
        module_path = option_mapping.get(selected_option)
        try:
            integration_module = importlib.import_module(module_path)
            return integration_module
        except ImportError as error:
            print(error)
            st.error(f"Failed to load the module for {selected_option}. {error}")
            return None
    else:
        st.error("Invalid option selected.")
        return None
    
if __name__ == "__main__":
    main()
