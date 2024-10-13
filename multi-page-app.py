import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import importlib
from onedrive_kb.main import start_indexing_thread
from config import Config
from streamlit_msal import Msal
from core.files.confluence_store import CustomVectorStore
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content .element-container {
        padding: 10px;
    }
    .sidebar .sidebar-content .element-container .stRadio {
        color: #4b4b4b;
        font-size: 18px;
    }
    .sidebar .sidebar-content .element-container .stRadio div {
        padding: 5px;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .sidebar .sidebar-content .element-container .stRadio div:hover {
        background-color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

api_key_placeholder = "********"

config = Config()
auth_data = None

def load_integration(selected_option):
    option_mapping = {
        "Ask AI": "generic_search.main",
        "Document AI": "document_ai.main",
        "Knowledge Base": "confluence_kb.main_new",
        "API Integration": "api_integration.main",
        "Text to SQL":"text_to_sql.main",
        "OneDrive":"onedrive_kb.main_v1",
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

def azure_sign_in():
    client_config = config.get_azure_credentials()
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

def clear_textbox():
    st.session_state.question = "" 
    st.session_state.query = "" 

def main():
    
    if "question" not in st.session_state:
        st.session_state["question"] = None

    if "query" not in st.session_state:
        st.session_state["query"] = None

    selected = option_menu(None, ['Home', 'T&T Visibility', 'Knowledge ChatBot', 'Operational Excellence', 'Concession Analyst'], 
        icons=['home', 'building-lock', 'box-seam-fill', 'award-fill', 'file-earmark-pdf-fill'], menu_icon="cast", default_index=0, orientation="horizontal")
   


    if "question" not in st.session_state:
        st.session_state.question = "" 
    if "query" not in st.session_state:
        st.session_state.query = "" 

    if 'current_page' not in st.session_state or st.session_state.current_page != selected:
        clear_textbox()
        st.session_state.current_page = selected 

    if selected == "Home":
        welcome()
    elif selected == "Concession Analyst":
        concessionMgmt()
    elif selected == "T&T Visibility":
        actionableInsights()
    elif selected == "Operational Excellence":
        opsExcellence()
    elif selected == "Knowledge ChatBot":
        knowledgeChatbot()

 

def welcome():
    # st.header("_Beyond the Hype:_ :blue[Real Examples of GPT's Power for Maersk and APMT] :sunglasses:")
    # st.subheader("Use Cases Covered")
    # st.markdown("- :red[Intelligent Concession Analyst] :green[GPT for our sales and commercial team(e.g. tell when what all terminals contract are up for renewal in next 5 years across Asia Middle East)]")
    # st.markdown("- :red[Intelligent T&T Visibility] :green[for our land, sea and rail side customer (e.g. where is my container)]")
    # st.markdown("- :red[Operational Excellence] :green[GPT for our terminal operations team  (e.g. give me an optimised berth planning solution with container lift per hour as key KPI)]")
    # st.markdown("- :red[Knowledge ChatBot] :green[for our internal employees (e.g. tell me more about APMT Strategy 2030)]")

    # st.markdown('''
    # <style>
    # [data-testid="stMarkdownContainer"] ul{
    #     padding-left:40px;
    # }
    # </style>
    # ''', unsafe_allow_html=True)

    image = Image.open('./images/new-theme-beyond-the-hype.jpg')
    new_image = image.resize((1440, 675))
    st.image(new_image)


def concessionMgmt():
    image = Image.open('./images/concession-mgmt.jpeg')
    new_image = image.resize((400, 200))
    left_co, cent_co,last_co = st.columns(3)
    with left_co:
        st.image(new_image, caption='Intelligent Concession Analyst', use_column_width="never")

    with last_co: 
        option = st.selectbox("Select a LLM Model",
        ("llama3.1", "OpenAI"),)
        st.write("You selected model:", option)

        source_option = st.selectbox(
        "Select a source for documentbase",
        ("ConcessionAnalyst"),
        )
        st.write("You selected documentbase:", source_option)
    

    st.session_state['model_name']=option
    integration_module = load_integration("Document AI")
    if integration_module:
        integration_module.run()

def opsExcellence():
    image = Image.open('./images/operational-excellence.jpeg')
    new_image = image.resize((400, 200))
    left_co, cent_co,last_co = st.columns(3)
    with left_co:
        st.image(new_image, caption='Operational Excellence', use_column_width="never")

    with last_co:   
        option = st.selectbox(
        "Select a LLM Model",
        ("llama3.1", "OpenAI"),
        )
        st.write("You selected model:", option)

    st.session_state['model_name']=option
    integration_module = load_integration("Text to SQL")
    if integration_module:
        integration_module.run()


def actionableInsights():
    image = Image.open('./images/actionable-business-insights.jpeg')
    new_image = image.resize((400, 200))
    left_co, cent_co,last_co = st.columns(3)
    with left_co:
        st.image(new_image, caption='T&T Visibility', use_column_width="never")

    with last_co:           
        option = st.selectbox(
        "Select a LLM Model",
        ("llama3.1"),
        )
        st.write("You selected model:", option)    

    st.session_state['model_name']=option
    integration_module = load_integration("API Integration")
    if integration_module:
        integration_module.run()

def knowledgeChatbot():
    image = Image.open('./images/employee-knowledgebase-chatbot.jpeg')
    new_image = image.resize((400, 200))
    left_co, cent_co,last_co = st.columns(3)
    with left_co:
        st.image(new_image, caption='Knowledge ChatBot', use_column_width="never")

    with last_co: 
        option = st.selectbox("Select a LLM Model",
        ("llama3.1", "OpenAI"),)
        st.write("You selected model:", option)

        source_option = st.selectbox(
        "Select a source for knowledgebase",
        ("Confluence", "OneDrive"),
        )
        st.write("You selected knowledgebase:", source_option)
    
    st.session_state['model_name']=option
    if source_option=="Confluence":
        integration_module = load_integration("Knowledge Base")
        if integration_module:
            integration_module.run()
    elif source_option=="OneDrive":
        manage_api_key()
        onedrive_kb_integration()
        integration_module = load_integration("OneDrive")
        if integration_module:
            integration_module.run()


if __name__ == "__main__":
    main()
