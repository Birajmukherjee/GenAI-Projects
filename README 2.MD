
# AI Knowledge Base Application with LLaMA and Streamlit

This project provides a comprehensive AI Knowledge Base Application combining various functionalities, 
such as NLP-driven API generation,
feedback analysis, 
document retrieval, 
knowledge base, and user authentication. 

It's powered by a locally running LLaMA model and allows for seamless user interactions through Streamlit.

---

## Core Functionalities

1. **Azure Sign-In and Authentication**:
    - Integrate Azure login functionality for authentication.

2. **API Query Generation**:
    - Use LLaMA to dynamically generate API endpoints and execute them.
    - Extract and fetch data from APIs using queries like “Get import availability for terminal esalg container UESU2254701.”

3. **Indexing and Document Retrieval**(Graph API Key required):
    - Automatic indexing of files from OneDrive. 
    - Search documents based on queries using LLaMA-powered AI.

4. **LLM-Backed Knowledge Base**(Uses logged in user token):
    - A knowledge base using LLaMA for answering user queries from Confluence data.

5. **Real-Time Streaming Responses**:
    - Display API call progress in real-time using streaming responses from the LLaMA model.

6. **Feedback Submission**:
    - Provide an appealing feedback form with ratings and comments submission features.

7. **Feedback Analysis**:
    - Use sentiment analysis to analyze user feedback stored in CSV format.
    - Provide visualizations such as pie charts, line charts, and word clouds.

---

## Prerequisites

1. **Python 3.12.6+**
2. **`pip` package manager**
3. **Ollama**: LLaMA's model manager for running the LLaMA models locally on macOS or Windows.
4. **Streamlit**: For the user interface.
5. **API access**: Ensure that you have API access (e.g., APM Terminals API).
6. **Azure Authentication**: Ensure you have the correct Azure app credentials for login integration.

---

## Installation and Setup

### Step 1: Clone the Repository

```sh

git clone https://github.com/Maersk-Global/apmt-llm-knowledge-base.git

cd apmt-llm-knowledge-base

```

### Step 2: Set Up Python Virtual Environment

```sh

python3 -m venv .venv

source .venv/bin/activate    
# On Windows: venv\Scripts\activate

```

### Step 3: Install Dependencies

```sh

pip install -r requirements.txt

```

### Step 4: Install and Run Ollama
https://github.com/ollama/ollama

#### For macOS:
Ollama allows running models locally and helps manage LLaMA models.

1. **Install Ollama** using Homebrew:

   ```sh
   
   brew install ollama
   
   ```

2. **Run Ollama**:

   ```sh
   
   ollama serve
   
   ```

3. **Download and Setup Models**:

   ```sh
   
   ollama pull llama3.1
   
   ```

4. Ensure the LLaMA model is running locally with the Ollama service listening at
   ```
   curl http://localhost:11434/api/tags 
   ```
   **POST Example**
   ```
   curl http://localhost:11434/api/generate -d '{
   "model": "llama3.1",
   "prompt":"Why is the sky blue?"
   }'
   ```

#### For Windows:

1. Download and install the Windows version of Ollama from the [official site](https://ollama.com/windows).
2. Follow the setup instructions to ensure the model is running locally.

---

## Running the Application

Once all the dependencies are installed, you can run the Streamlit app locally:

```bash

streamlit run app.py

```

This will start the local development server, 
and you can access the application in your browser at
`http://localhost:8501`.

---

## How to Use the Application

### 1. **API Query Generation**:
- Input natural language questions like "Get import availability for terminal esalg container UESU2254701."
- The system generates API endpoints, makes the API calls, and displays the response.

### 2. **Feedback Analysis**:
- Upload a CSV file containing user feedback (columns: Name, Role, Rating, Comments).
- The system analyzes the feedback using sentiment analysis and visualizes the data using pie charts, bar charts, and word clouds.

### 3. **Document Retrieval and Indexing**:
- Upload documents from OneDrive or Confluence. 
- To get the Graph API token (https://developer.microsoft.com/en-us/graph/graph-explorer), click on "Access Token"
- Search specific content using LLaMA-powered natural language queries.

### 4. **Azure Sign-In**:
- Users can sign in with their Azure credentials to access personalized services.

### 5. **Knowledge Base**:
- Ask questions from indexed documents using LLaMA as the AI backend for real-time responses.

### 6. **Real-Time Streaming**:
- View live streaming of responses from LLaMA when processing user queries or generating API calls.

---

## Folder Structure

- `app.py`: The main Streamlit app integrating various features like API generation, feedback analysis, and document retrieval.
- `api_integration.py`: Handles API documentation for querying APM Terminals' APIs.
- `feedback_analysis.py`: Handles sentiment analysis, visualizations, and word cloud generation.
- `onedrive_kb.py`: OneDrive integration for document indexing and search.
- `azure_login.py`: Azure login functionality.
- `config.py`: Handles configuration and environment variables.
  - Make sure all environment variables are set in local before running the application
    - Mac : export OPEN_API_KEY=''
- `requirements.txt`: All dependencies required for the project.

---

## Additional Functionality

### 1. **Feedback System**: 
   - Capture and visualize user feedback, including ratings and comments.
### 2. **Authentication**: 
   - Azure login functionality with personalized experiences.
### 3. **Threading for Indexing**: 
   - Multi-threaded file indexing for continuous updates from OneDrive or Confluence.
### 4. **LLM Integration**: 
   - Leverage LLaMA as the primary NLP model for natural language queries.
---

## Customization and Future Enhancements

1. **Adding New APIs**:
    - Extend `api_integration.py` with new API documentation and corresponding functionality.

2. **Custom Feedback Analytics**:
    - Modify the feedback analysis module to add custom analysis features like more advanced sentiment classification or topic modeling.

3. **More Authentication Options**:
    - Integrate OAuth2 or SSO mechanisms for a more flexible login process.

---

## Known Issues and Debugging

1. **Connection Errors**:
    - Ensure that the LLaMA model is running locally, and the API URL is properly formatted.

2. **Ollama Model Not Running**:
    - Verify that Ollama is installed and running the required models using the command:

      ```bash
      
      ollama serve
      
      ```

3. **Streaming Issues**:
    - If the Streamlit app doesn't show real-time updates, ensure you're running Streamlit with `stream=True` for the LLaMA API.

---

## Testing

- **Unit Tests**: Write unit tests for each of the core modules (API generation, feedback analysis, document retrieval). (PENDING)
- **Integration Tests**: Test the app's integration with LLaMA and external APIs. (PENDING)
- **UI Testing**: Test the Streamlit interface for proper responsiveness and user interactions. (PENDING)

---