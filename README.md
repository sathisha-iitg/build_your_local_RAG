# 📝 Build Your Local RAG System with LLMs

Welcome to the **Local LLM-based Retrieval-Augmented Generation (RAG) System**! This repository provides the full code to build a private, offline RAG system for managing and querying personal documents locally using a combination of OpenSearch, Sentence Transformers, and Large Language Models (LLMs). Perfect for anyone seeking a privacy-friendly solution to manage documents without relying on cloud services.


### 🌟 Key Features:
- **Privacy-Friendly Document Search:** Search through personal documents without uploading them to the cloud.
- **Hybrid Search with OpenSearch:** Uses both traditional text matching and semantic search.
- **Easy Integration with LLMs**: Leverage local LLMs for personalized, context-aware responses.

### 🚀 Get Started
1. Clone the repo: `git clone `
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `constants.py` for embedding models and OpenSearch settings.
4. Run the Streamlit app: `streamlit run welcome.py`

### Detailed Installation Guid
# 🚀 Local RAG System with OpenSearch, PyTesseract, and Ollama

This repository walks you through building a **local Retrieval-Augmented Generation (RAG) system** from scratch. You’ll learn how to set up Docker, configure OpenSearch, integrate PyTesseract for OCR, and generate embeddings with SentenceTransformers. By the end, you’ll have a fully functional local RAG system that can index and query your own documents.

---

## 🛠️ Prerequisites

Before running the application, install and configure the following tools:

### 1. Install Docker
Docker lets us run OpenSearch locally in an isolated environment.  

- [Install Docker](https://docs.docker.com/engine/install/)  
- Verify installation:
  ```bash
  docker --version
  ```

### 2. Install Ollama
Ollama makes it simple to run language models locally.  

- [Download Ollama](https://ollama.com/download)  
- Verify installation:
  ```bash
  ollama --version
  ```
- Test with a model:
  ```bash
  ollama run llama3.2:1b
  ```

### 3. Set Up OpenSearch and Dashboard
OpenSearch acts as the **vector database** for storing embeddings.  

- Pull Docker images:
  ```bash
  docker pull opensearchproject/opensearch:2.11.0
  docker pull opensearchproject/opensearch-dashboards:2.11.0
  ```
- Run OpenSearch:
  ```bash
  docker run -d --name opensearch     -p 9200:9200 -p 9600:9600     -e "discovery.type=single-node"     -e "DISABLE_SECURITY_PLUGIN=true"     opensearchproject/opensearch:2.11.0
  ```
- Run Dashboard:
  ```bash
  docker run -d --name opensearch-dashboards     -p 5601:5601     --link opensearch:opensearch     -e "OPENSEARCH_HOSTS=http://opensearch:9200"     -e "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true"     opensearchproject/opensearch-dashboards:2.11.0
  ```
- Visit [http://localhost:5601](http://localhost:5601) to confirm setup.

### 4. Enable Hybrid Search in OpenSearch
Hybrid search blends BM25 and vector-based search for better results.  

- Create a search pipeline:
  ```bash
  curl -XPUT "http://localhost:9200/_search/pipeline/nlp-search-pipeline"     -H 'Content-Type: application/json' -d'
  {
    "description": "Post processor for hybrid search",
    "phase_results_processors": [
      {
        "normalization-processor": {
          "normalization": {
            "technique": "min_max"
          },
          "combination": {
            "technique": "arithmetic_mean",
            "parameters": {
              "weights": [0.3, 0.7]
            }
          }
        }
      }
    ]
  }'
  ```

Or paste the JSON into **OpenSearch Dashboard → Dev Tools**.

### 5. Install Python 3.11 and Set Up Virtual Environment
We use Python 3.11 for this project.  

- Verify installation:
  ```bash
  python3.11 --version
  ```
- Create a virtual environment:
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

---

## ⚡ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sathisha-iitg/build_your_local_RAG.git
cd <repo-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- Streamlit (UI)
- SentenceTransformers (embeddings)
- PyTesseract (OCR)
- and other required libraries.

### 3. Configure Constants
Edit `src/constants.py` to match your setup:

- **EMBEDDING_MODEL_PATH**  
  Path or Hugging Face model name (e.g. `"sentence-transformers/all-mpnet-base-v2"`).  
- **EMBEDDING_DIMENSION**  
  `768` for `all-mpnet-base-v2`, `384` for `all-MiniLM-L12-v2`.  
- **TEXT_CHUNK_SIZE**  
  Recommended: `300`.  
- **OLLAMA_MODEL_NAME**  
  Example: `"llama3.2:1b"`.

### 4. Run the Application
```bash
streamlit run welcome.py
```

- Open [http://localhost:8501](http://localhost:8501)  
- Wait for models to load/download on first run.  

---

## 🎉 Features

- **📄 Document Upload**: Drag and drop PDFs for OCR, text chunking, and embedding generation.  
- **🔎 Hybrid Search**: Combines BM25 and vector similarity for accurate retrieval.  
- **🤖 Chatbot Interface**: Ask natural language questions about your documents.  
- **⚡ Retrieval-Augmented Generation**: Enable RAG mode to inject document context into LLM responses.  
- **⚙️ Customizable Settings**: Adjust top search results, model, chunk size, and temperature.  

---

## ✅ You’re All Set!
With these steps, your **local RAG system** is ready to handle document search and Q&A.  
Upload documents, interact with the chatbot, and experiment with settings to explore the system’s full potential.
