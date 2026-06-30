# src/retrieval.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def verify_retrieval():
    persistent_db_directory = "./db"
    collection_name = "gpu_specs_collection"
    
    print("[1/3] Re-initializing HuggingFace Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("[2/3] Connecting to persistent Chroma database")
    # Load the existing DB from disk
    vector_db = Chroma(
        persist_directory=persistent_db_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Define a test question relevant to your data
    test_query = "What is the memory capacity and bandwidth of the NVIDIA Blackwell B200?"
    print(f"\n[3/3] Issuing test query to DB: '{test_query}'")
    
    # Retrieve top 3 matching chunks
    docs = vector_db.similarity_search(test_query, k=3)
    
    print("\n================== RETRIEVED CONTEXT FRAGMENTS ==================")
    if not docs:
        print(" CRITICAL: No matching documents retrieved. Check collection name or chunk sizes!")
    else:
        for idx, doc in enumerate(docs):
            print(f"\n--- Fragment {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')}) ---")
            print(doc.page_content)
    print("================================================================")

if __name__ == "__main__":
    verify_retrieval()