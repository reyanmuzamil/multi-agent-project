
# System & Operational Tools
import os
from dotenv import load_dotenv

# LLM Core Components (LangChain Ecosystem)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Shared Project State Architecture Contract
from state import AgentState

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("GROQ_API_KEY")

def writer_agent(state: AgentState) -> dict:
    user_query = state.get("user_query", "")
    research_data = state.get("research_data", [])
    vetted_facts = state.get("vetted_facts", [])
    
    print("\n[Writer Agent]  Generating definitive technical analysis report...")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3, 
        groq_api_key=api_key
    )
    
    system_instruction = (
        "You are an expert technical writer and industry analyst specializing in hardware metrics and market intelligence.\n\n"
        "Your task is to compile a definitive, publication-ready technical analysis report answering the user's primary inquiry.\n\n"
        "You will be provided with:\n"
        "- The original broad user query.\n"
        "- The raw research context streams compiled for each investigated sub-question.\n"
        "- A curated, verified list of claims evaluated by our Fact-Checker Agent containing SUPPORTED, CONTRADICTED, or UNVERIFIED truth statuses.\n\n"
        "You must structure your Markdown report exactly using the following structural blueprint:\n"
        "## 1. Executive Summary\n"
        "A high-level synthesis addressing the user query based on confirmed evidence.\n\n"
        "## 2. Technical Deep-Dive & Comparison Matrices\n"
        "Synthesize the research data. Whenever technical metrics, specifications, or benchmarks are mentioned, "
        "you MUST format them inside clean Markdown Comparison Tables for readable presentation.\n\n"
        "## 3. Fact Audit & Grounding Ledger\n"
        "Generate a clear tracking table detailing the truth audits performed on claims in the system. Use this structure:\n"
        "| Extracted Claim | Truth Status | Auditor Analytical Reasoning |\n"
        "|---|---|---|\n"
        "Ensure all provided vetted facts are cleanly listed here so the user can easily trace supported vs contradicted statements.\n\n"
        "Maintain a neutral, highly objective, corporate tone throughout. Do not mention agent names like 'Fact-Checker Agent' or 'Researcher Agent' in your final text—present this as a unified technical output."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        (
            "user", 
            "PRIMARY USER QUERY: {query}\n\n"
            "ACCUMULATED RESEARCH BLOCKS:\n{research}\n\n"
            "VETTED FACTS & VERDICTS REGISTER:\n{facts}"
        )
    ])
    
    chain = prompt | llm
    
    # Invoke the chain using the extracted state attributes
    response = chain.invoke({
        "query": user_query,
        "research": str(research_data),
        "facts": str(vetted_facts)
    })
    
    print("[Writer Agent]  Technical report generated successfully.")
    
    # Return dictionary to update the graph whiteboard state contract
    return {"final_report": response.content}

# --- QUICK ISOLATION UNIT TEST ---
if __name__ == "__main__":
    # Fake state simulating a successful upstream run
    test_state: AgentState = {
        "user_query": "NVIDIA Blackwell metrics.",
        "sub_questions": ["What is the memory bandwidth of NVIDIA Blackwell?"],
        "research_data": [{"sub_question": "What is the memory bandwidth?", "local_rag_context": "8.0 TB/s", "web_search_context": "", "news_context": ""}],
        "vetted_facts": [
            {"claim": "Memory bus hits 8.0 TB/s throughput natively.", "verdict": "SUPPORTED", "reasoning": "Confirmed via local RAG metadata and media wires."},
            {"claim": "Blackwell hits 10 TB/s rumors.", "verdict": "CONTRADICTED", "reasoning": "Contradicted by local specifications sheet."}
        ],
        "final_report": ""
    }
    
    result = writer_agent(test_state)
    print("\n======================= OUTPUT MARKDOWN REPORT =======================")
    print(result["final_report"])