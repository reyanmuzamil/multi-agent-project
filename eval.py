import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

from main import multi_agent_system
from state import AgentState

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

agent_app = multi_agent_system()

def target_runner(inputs: dict) -> dict:
    time.sleep(5)
    
    initial_state: AgentState = {
        "user_query": inputs["user_query"],
        "sub_questions": [],
        "research_data": [],
        "vetted_facts": [],
        "final_report": ""
    }
    
    final_output_state = {}
    for event in agent_app.stream(initial_state):
        for node_name, state_update in event.items():
            final_output_state = state_update
            
    return {"output": final_output_state.get("final_report", "")}

def run_automated_eval_suite():
    dataset_name = "Hardware-Benchmark-Dataset"
    print(f" Initializing automated LangSmith evaluation against dataset: '{dataset_name}'...")
    
    eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=api_key)
    
    qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm})
    context_qa_evaluator = LangChainStringEvaluator("context_qa", config={"llm": eval_llm})
    
    experiment_results = evaluate(
        target_runner,
        data=dataset_name,
        evaluators=[qa_evaluator, context_qa_evaluator], 
        experiment_prefix="multi-agent-v1-test"
    )
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    run_automated_eval_suite()