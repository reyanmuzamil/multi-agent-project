import os
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Modern Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq 

load_dotenv()

def run_ingestion():
    # File reading 
    raw_data_path = "data.txt"
    persistent_db_directory = "./db" # This name must match what Chroma uses below
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Source file not located at {raw_data_path}. Please create it.")
        
    print("1/4 Reading the file...")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # Chunking 
    print("2/4 Now chunking the data...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   
        chunk_overlap=80, 
    )
    chunks = splitter.split_text(raw_text)
    
    doc_to_insert = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,  
            metadata={"source": "data.txt", "chunk_index": i}, 
        )
        doc_to_insert.append(doc) 
        
    print(f"Successfully prepared {len(doc_to_insert)} documents for ingestion.")
    
    # Importing hugging face model 
    print("3/4 Initializing HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("4/4 Vectorizing and saving documents to local database...")
    vector_db = Chroma.from_documents(
        documents=doc_to_insert,               
        embedding=embeddings,                     
        persist_directory=persistent_db_directory,
        collection_name="gpu_specs_collection"
    )
    print("Database finalized and saved securely inside './db/' directory!")

if __name__ == "__main__":
    run_ingestion()