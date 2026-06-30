import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from dotenv import load_dotenv

# BULLETPROOF ENVIRONMENT LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")

def planner_agent(state: AgentState) -> dict:
    #doc string 
    """
    Takes the raw user_query from the state, breaks it down into 
    2-4 specific sub-questions, and updates the state. 
    Includes defensive API debugging metrics to avoid catastrophic crashes.
    """
    print("\n[Planner Agent] Analyzing global query and orchestrating research sub-questions...")
    user_query = state.get("user_query", "")
    
    if not user_query:
        print("[Planner Agent] Warning: Empty user query received.")
        return {"sub_questions": []}
        
    # --- ENVIRONMENT CHECKS ---
    if not api_key:
        print(" CRITICAL ERROR: GROQ_API_KEY could not be found in your local .env configuration.")
        return {"sub_questions": []}

    try:
        # Modern explicit initialization using an active production model
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  
            temperature=0.1,
            groq_api_key=api_key  
        )
        
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technical planner specializing in semiconductor architectures, AI hardware, and tech market analytics.\n\n"
                "Your task is to take a complex user query and break it down into 2 to 4 distinct, atomic sub-questions.\n"
                "These sub-questions will be passed directly to research tools to gather data on technical specifications, metrics, or current market news.\n\n"
                "CRITICAL: You must return ONLY a raw JSON array of strings. Do not include any introductory text, markdown formatting blocks (like ```json), or explanatory footnotes.\n"
                "Example Output:\n"
                "[\"What are the specific FP4 performance metrics of NVIDIA Blackwell?\", \"What is the peak memory bandwidth of the AMD MI300X?\"]"
            )),
            ("user", "Deconstruct this query into research sub-questions: {query}")
        ])
        
        chain = planner_prompt | llm
        
        # Safe API invocation block
        response = chain.invoke({"query": user_query})
        raw_content = response.content.strip()
        
    except Exception as api_err:
        print(f"\n GROQ API CALL FAILED. Diagnostic message:\n{api_err}\n")
        # Graceful fallback return to keep the LangGraph pipeline from locking up
        return {"sub_questions": []}
    
    # Defensive JSON clean-up block in case the LLM wrapped it in markdown code fences
    if raw_content.startswith("```json"):
        raw_content = raw_content.split("```json")[1].split("```")[0].strip()
    elif raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1].split("```")[0].strip()
        
    try:
        sub_questions = json.loads(raw_content)
        if not isinstance(sub_questions, list):
            sub_questions = [str(sub_questions)]
        print(f"[Planner Agent] Successfully planned {len(sub_questions)} sub-questions.")
    except Exception as parse_err:
        print(f"[Planner Agent] JSON Parsing Failed. Falling back to string splits. Reason: {parse_err}")
        # Secondary backup parser logic
        sub_questions = [q.strip() for q in raw_content.replace("[", "").replace("]", "").replace('"', '').split(",") if q.strip()]

    return {"sub_questions": sub_questions}


# --- QUICK ISOLATION UNIT TEST ---
if __name__ == "__main__":
    test_state: AgentState = {
        "user_query": "Compare the memory bandwidth and FP8 performance between NVIDIA Blackwell and AMD Instinct MI300X based on specs.",
        "sub_questions": [],
        "research_data": [],
        "vetted_facts": [],
        "final_report": ""
    }
    
    result = planner_agent(test_state)
    print("\nReturned Mutation state:", result)