# RAG-Assistant
**This is a pure text based RAG Assistant that answers questions on IT Policy,HR Rules of a fictional organization
   The docs folder consists of 2 documents-> IT security policy is a Text file that consists of 
**
# Tech Stack Used
** 1. Programming: Python
** 2. Container: Dockers
** 3. Vector Search: ChromaDB
** 4. LLM Framework: Langchain
** 5. Model: OpenAI

# Langchain COncepts Used:
** 1. Langchain Prompt Templates
** 2. TextLoader for chunking through Text documents
** 3. OpenAI Embeddings: text-embedding-3-small
** 4. LCEL Based Chaining
   
# Docker Setup
**1. Using terminal type docker compose up -d
**2. Change status of the container using docker ps
**3. To Check the status of the chromadb , use this url: http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections
# Run the app
**1. Run the script using python rag_script_1.py
**2. The program would ask for a prompt or press q to continue. (Note:This is a text based app so the interface is through the command line interface)
**3. On the prompt asked, the 
