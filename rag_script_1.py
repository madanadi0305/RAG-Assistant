import os
from pydoc import doc
import time
# from chromadb.types import C  # ❌ Removed: unused and invalid import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# from langchain_core.runnables import Runnable  # ❌ Removed: unused import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # ❌ Removed unused: OpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
# from openai.types import vector_store  # ❌ Removed: conflicts with local variable name
load_dotenv()
logging.basicConfig(filename='master_logs.logs',level=logging.INFO,format='%(asctime)s[%(levelname)s] %(name)s - %(message)s',datefmt='%Y-%m-%d %H-%M-%S')
chromadb_client = None
path ="C:\\Users\\madan\\Downloads\\RAG Project\\docs\\"
port = int(os.environ["PORT"])          # ✅ Cast to int (HttpClient requires int port)
server_name = os.environ["SERVER_NAME"]
dir_path = path                         # ✅ Fixed: os.path.dirname() strips the last folder,
                                        #    use path directly as the docs directory
logging.debug("RAG Application")

if not os.environ.get('OPENAI_API_KEY'):  # ✅ Use .get() to avoid KeyError if missing
    
    logging.error("API Key Not found")
else:
   
    logging.info("Environment defined successfully")

@retry(stop=stop_after_attempt(5),wait=wait_exponential(multiplier=1,max=2,min=10))
def instantiate_chroma_server():
    global chromadb_client
    logging.info("Instantiating connection with chromadb")
    try:
        chromadb_client = chromadb.HttpClient(host=server_name, port=port)
        logging.info("Chroma server initialized")
        # ✅ Fixed: keyword arg is `host`, not `str`
    except Exception as e:
        logging.critical("Error connecting with Chromadb Client: %s", e)
        # ✅ Fixed: logging.error() doesn't accept a `msg` keyword arg
     
        raise
def retrieve_document_from_docstore(file_name:str,collection_name:str):
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    vs=Chroma(collection_name=collection_name,embedding_function=embeddings,client=chromadb_client)
    file_result=vs.get(where={"source":file_name})
    
    if  file_result or len(file_result.get("documents",[]))>0:
        stored_timestamp = file_result["metadatas"][0].get("last_modified", 0)
        return True,stored_timestamp
    return False,0
    
def ingest_documents():
    logging.debug("Ingesting documents...")
    collection_name=os.environ["COLLECTION_NAME"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    existing_collections=chromadb_client.list_collections()
    print("Existing:",existing_collections)
    if collection_name in  [c.name for c in existing_collections]:
        logging.info("Collection already exists.Can't ingest documents..")
        vs=Chroma(collection_name="it_hr_policy_rag",embedding_function=embeddings,client=chromadb_client)
        
    #Check if documents exist or not:
      
    all_chunks = []
    metadata_registry = {
        "hr_policy.txt":      {"department": "HR",        "security_level": "public"},
        "it_security_protocol.txt": {"department": "IT",        "security_level": "internal"},
       # "q3_marketing_plan.pd":   {"department": "Marketing", "security_level": "confidential"}
    }
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )

    for doc in os.listdir(dir_path):
        
        full_path = os.path.join(dir_path, doc)   # ✅ Fixed: build full file path
        #Check if document exists or not
        doc_name=str(full_path)
        local_timestamp=os.path.getmtime(full_path)
        result,doc_timestamp=retrieve_document_from_docstore(file_name=full_path,collection_name=collection_name)
        if result:
            #Check if update was made or not
                    
            if doc_timestamp<local_timestamp:
                #File still is the same
                logging.info("Deleting existing chunks and uploading new")
                vs.delete(where={"source":full_path})
            else:
                logging.info("Document already upto date no need to do  anything")
                continue
        logging.info("Creating document")

        loader_file_instance = TextLoader(file_path=full_path)
        logging.info("Extracting Document", doc)
        
        pages = loader_file_instance.load()
        chunks = text_splitter.split_documents(pages)

        custom_obj = metadata_registry.get(doc, {"department": "General", "security_level": "public"}).copy()
        custom_obj["source"]=full_path
        custom_obj["last_modified"]=local_timestamp
        # ✅ Fixed: metadata loop was outside the for-doc loop (wrong indentation)
        for chunk in chunks:
            chunk.metadata.update(custom_obj)

        logging.debug("Chunks appended")
        all_chunks.extend(chunks)   # ✅ Fixed: moved inside the for-doc loop

    if not all_chunks:
        logging.debug("Chunks not appended")
        return None   # ✅ Explicit None return so callers can check safely

    logging.info("Chunks appended successfully")

    
    if not chromadb_client:
        return None
    else:
        vs = Chroma(                  # ✅ Renamed from vector_store to avoid shadowing
            collection_name="it_hr_policy_rag",
            embedding_function=embeddings,
            client=chromadb_client,
            #collection_metadata={"hnsw:space":"cosine"}
        )
        vs.add_documents(documents=all_chunks) 
        logging.info("Chunks added successfully")  # ✅ Fixed: use all_chunks, not chunks
        return vs
    


def retriever_logic(vs,user_question: str):
   # vs = ingest_documents()

    if not vs:   # ✅ Added: guard against None before calling .as_retriever()
        logging.error("Vector store unavailable; cannot retrieve.")
        return

    vector_retriever = vs.as_retriever()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant who gathers information from the documents and summarizes it.
        Answer only using the provided context
        If the information is not avaialble in the documents, respond by saying
        I don't have enough relevant information
        "Keep everything professional,precise and accurate"
        """),
        ("human", "Context {context}{question}")
    ])

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    output_parser_obj = StrOutputParser()   # ✅ Fixed: JsonOutputParser won't work here
                                            #    unless the LLM is explicitly prompted for JSON.
                                            #    StrOutputParser is the correct default.

    # ✅ Fixed: added RunnablePassthrough so the retriever result is passed into the prompt
    

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | output_parser_obj
    )

    result = chain.invoke(user_question)   # ✅ Fixed: must pass the question argument
    #print("Result:", result)
    html_resp=f"""<html>
    <body>
    <head>Assistant Response</head>
    <p>
    {result}
    </p>
    </body>
    </html> """
    print(html_resp)
    return html_resp


def main():
    logging.debug("Instantiating Application Server...")
    instantiate_chroma_server()
    vs=ingest_documents()
    print(vs)
    if not vs:
        logging.debug("Vector Store not defined,Failed to initialize")
        return

    while True:
        user_input=input("Provide a query. Press c to continue or q to quit")
        
        user_query=str(user_input)
        if user_input=="q":
            break 
        retriever_logic(vs,user_question=user_query)
        


if __name__ == "__main__":   # ✅ Fixed: must be "__main__", not "main"
    main()