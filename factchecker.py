
# System & Operational Tools
import os
import json
from typing import Dict, Any, List

# Environment Configuration
from dotenv import load_dotenv

# LLM Core Components (LangChain Ecosystem)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

 #Shared Project State Architecture Contract
from state import AgentState
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("GROQ_API_KEY")
def factchecker_agent(state:AgentState)->dict:
    llm=ChatGroq(model="llama-3.3-70b-versatile",
                temperature=0.1,
                 groq_api_key=api_key)
    # Updated system instruction with escaped curly braces
    system_instruction = (
        "You are an elite, highly objective quality assurance judge and technical auditor.\n\n"
        "Your task is to extract core technical claims answering a given sub-question and evaluate them "
        "strictly against the provided source text context markers (RAG, Web, and News data).\n\n"
        "Follow these 3 absolute execution rules:\n"
        "1. Analyze the sub-question and extract explicit technical, metric-driven, or market-facing claims.\n"
        "2. Inspect the provided source contexts to cross-examine each claim and issue a precision truth verdict.\n"
        "3. Output ONLY a raw, un-markdowned JSON array of objects. Do not include introductory text, conversational fluff, or markdown code blocks (like ```json).\n\n"
        "Every single object inside the JSON array MUST follow this identical schema:\n"
        "{{\n"  # Double curly brace to escape
        '  "claim": "The exact factual statement or metric value being verified.",\n'
        '  "verdict": "MUST be exactly one of these uppercase tokens: SUPPORTED, CONTRADICTED, or UNVERIFIED",\n'
        '  "reasoning": "A concise sentence validating your choice using only raw evidence found in the text."\n'
        "}}\n"  # Double curly brace to escape
    )
    all_vetted_facts=[]
    research_data=state.get("research_data",[])
    for item in research_data:
        sub_ques=item.get("sub_questions","")
        rag=item.get("local_rag_context","")
        web=item.get("web_search_context","")
        news=item.get("news_context","")
        context_block = f"DATA FROM LOCAL RAG:\n{rag}\n\nDATA FROM WEB SEARCH:\n{web}\n\nDATA FROM NEWS MEDIA:\n{news}"        
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_instruction),
                ("user","SUB-QUESTION BEING INVESTIGATED: {question}\n\nCONSOLIDATED EVIDENCE BASE:\n{evidence}"),
            ]
        )
        chain=prompt|llm
        try:
            response = chain.invoke({"question": sub_ques, "evidence": context_block})
            raw_content = response.content.strip()
            
            # --- DEFENSIVE JSON CLEANUP LOGIC ---
            if raw_content.startswith("```json"):
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1].split("```")[0].strip()
                
            # Safely parse string into an active Python list
            vetted_claims = json.loads(raw_content)
            
            if isinstance(vetted_claims, list):
                all_vetted_facts.extend(vetted_claims)
                print(f" -> [Fact-Checker] Successfully cross-examined {len(vetted_claims)} claims for: '{sub_ques}'")
            else:
                # If the LLM returned a single JSON object instead of an array, wrap it safely
                all_vetted_facts.append(vetted_claims)
                
        except Exception as e:
            print(f" -> [Fact-Checker] Parsing breakdown on sub-question: '{sub_ques}'. Error: {e}")
            # Inject a fallback item to avoid completely killing graph data tracking
            all_vetted_facts.append({
                "claim": f"Verification execution error for: {sub_ques}",
                "verdict": "UNVERIFIED",
                "reasoning": f"System engine failed to decode text context safely. Exception: {str(e)}"
            })
            
    print(f"\n[Fact-Checker Agent] Verification complete. Checked total of {len(all_vetted_facts)} assertions.")
    
    # Return key matches our AgentState dictionary contract
    return {"vetted_facts": all_vetted_facts}

# --- QUICK ISOLATION UNIT TEST ---
if __name__ == "__main__":
    test_state: AgentState = {
        "user_query": "Check bandwidth metrics.",
        "sub_questions": ["What is the memory bandwidth of NVIDIA Blackwell?"],
        "research_data": [
            {
                "sub_question": "What is the memory bandwidth of NVIDIA Blackwell?",
                "local_rag_context": "The memory sub-system provides a capacity of 192 Gigabytes running at a bus speed of 8.0 Terabytes per second (TB/s).",
                "web_search_context": "Some online forums claim Blackwell B200 hits 10 TB/s while others say 8 TB/s.",
                "news_context": "Press releases confirm Blackwell architectural memory configurations run natively up to 8.0 TB/s throughput rates."
            }
        ],
        "vetted_facts": [],
        "final_report": ""
    }
    
    result = factchecker_agent(test_state)
    print("\n================== VERIFIED OUTCOME MATRIX ==================")
    print(json.dumps(result, indent=2))




