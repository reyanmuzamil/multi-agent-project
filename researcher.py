#  System & Operational Tools
import os
import json
from typing import Dict, Any, List

#  Environment Configuration
from dotenv import load_dotenv
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

try:
    from tavily import TavilyClient
except ImportError:
    print("[Researcher Agent] Warning: Tavily library not found in local site-packages.")
    TavilyClient = None

import requests  # for news api 

# 4. Local Vector Database Infrastructure 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#Shared Project State Architecture Contract
from state import AgentState

# Pull explicit API settings
Tavily_Api_key = os.getenv("TAVILY_API_KEY")
T_API = TavilyClient(api_key=Tavily_Api_key) if (TavilyClient and Tavily_Api_key) else None
News_api = os.getenv("NEWSAPI_KEY")


def researcher_agent(state: AgentState) -> dict:
    """
    Reads planned sub-questions from the shared state, queries a local Chroma vector 
    store along with web components (Tavily and NewsAPI), and aggregates contextual payloads.
    """
    print("\n[Researcher Agent] Initiating multi-source factual discovery process...")
    
    # Connecting to the actual vector database using absolute directory resolution
    persistent_db_directory = os.path.join(base_dir, "db")
    collection_name = "gpu_specs_collection"
    
    try:
        #v=connecting to our vector database 
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(
            persist_directory=persistent_db_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
    except Exception as db_init_err:
        print(f" DB Initialization failure: {db_init_err}")
        vector_db = None
     
    # Getting the subquestions 
    questions = state.get("sub_questions", [])
    new_research_payloads = []
    for question in questions:
        print(f"\nProcessing Sub-Question: '{question}'")
        
        # --- PART A: Local Vector Database ---
        combined_local_context = ""
        if vector_db:
            try:
                docs = vector_db.similarity_search(question, k=2)
                local_text_fragments = []
                if not docs:
                    print("  Warning: No matching documents retrieved from vector space.")
                else:
                    for doc in docs:
                        local_text_fragments.append(doc.page_content)
                combined_local_context = "\n".join(local_text_fragments)
            except Exception as e:
                print(f"  Vector DB query failed: {e}")
                combined_local_context = "Local vector data search faulted."
        else:
            combined_local_context = "Vector database unavailable."
        
        # --- PART B: Web search using Tavily API ---
        combined_web_context = ""
        if T_API:
            try:
                results = T_API.search(query=question, max_results=2)
                web_text_fragments = []
                for item in results.get("results", []):
                    content_snippet = item.get("content", "")
                    if content_snippet:
                        web_text_fragments.append(content_snippet)
                combined_web_context = "\n".join(web_text_fragments)
                print(" -> [Tavily Search] Successfully collected live web data.")
            except Exception as e:
                print(f"  Tavily Search failed: {e}")
                combined_web_context = "No real-world web updates available."
        else:
            print("  Warning: Tavily Client unconfigured or key missing.")
            combined_web_context = "Tavily web tracking skipped."
        
        # --- PART C: News API Call ---
        combined_news_context = ""
        if News_api:
            try:
                print(" -> Requesting hot media feeds from NewsAPI...")
                url = "https://newsapi.org/v2/everything"
                params = {
                     "q": question,
                     "pageSize": 2,
                     "apiKey": News_api
                }
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    news_data = response.json()
                    news_text_fragments = []
                    
                    for article in news_data.get("articles", []):
                        title = article.get("title", "")
                        description = article.get("description", "")
                        if title or description:
                            news_text_fragments.append(f"Title: {title}\nSummary: {description}")
                               
                    combined_news_context = "\n\n".join(news_text_fragments)
                    print(" -> [NewsAPI] Successfully collected media records.")
                else:
                    print(f" NewsAPI request returned status code: {response.status_code}")
                    combined_news_context = "News data unavailable due to API status."
            except Exception as e:
                print(f" NewsAPI request failed: {e}")
                combined_news_context = "No live media tracking updates available."
        else:
            print("  Warning: NEWSAPI_KEY missing. News API tracking skipped.")
            combined_news_context = "News API Key not configured."

        # --- PACKAGING AND MERGING ---
        single_question_payload = {
             "sub_question": question,
             "local_rag_context": combined_local_context if combined_local_context else "No local specifications found.",
             "web_search_context": combined_web_context if combined_web_context else "No web search results.",
             "news_context": combined_news_context if combined_news_context else "No industry news found."
        }
        new_research_payloads.append(single_question_payload)
        print("===========================================================================")
        
    # Return the key to append to our LangGraph state contract
    return {"research_data": new_research_payloads}


# --- QUICK ISOLATION UNIT TEST ---
if __name__ == "__main__":
    test_state: AgentState = {
        "user_query": "Compare chip data.",
        "sub_questions": [
            "What is the memory bandwidth of NVIDIA Blackwell?",
            "What are the latest shipping updates for AMD Instinct MI300X?"
        ],
        "research_data": [],
        "vetted_facts": [],
        "final_report": ""
    }
    
    result = researcher_agent(test_state)
    print("\n================== RESEARCHER METRICS =================")
    print(f"Total structured payloads returned: {len(result['research_data'])}")