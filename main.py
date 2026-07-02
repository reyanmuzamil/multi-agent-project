import os
from dotenv import load_dotenv

# Import LangGraph Structural Core Elements
from langgraph.graph import StateGraph, START, END

# Import Shared Project State Architecture Contract
from state import AgentState

# Import Your Specialized Worker Agent Nodes
from planner import planner_agent
from researcher import researcher_agent
from factchecker import factchecker_agent
from writer import writer_agent

# Load Environmental Variables
load_dotenv()
def multi_agent_system():
    workflow=StateGraph(AgentState)
    workflow.add_node("Planner_Agent",planner_agent)
    workflow.add_node("Researcher_Agent",researcher_agent)
    workflow.add_node("FactChecker_Agent",factchecker_agent)
    workflow.add_node("Writer_Agent",writer_agent)

    workflow.add_edge(START,"Planner_Agent")
    workflow.add_edge("Planner_Agent","Researcher_Agent")
    workflow.add_edge("Researcher_Agent","FactChecker_Agent")
    workflow.add_edge("FactChecker_Agent","Writer_Agent")
    workflow.add_edge("Writer_Agent",END)
    app = workflow.compile()
    return app

def main():
    print("==================================================================")
    print(" MULTI-AGENT VERIFICATION ENGINE RUNTIME ACTIVE")
    print("==================================================================")
    
    # Compile the graph application
    agent_app = multi_agent_system()
    
    # Standard terminal interaction loop
    while True:
        try:
            user_input = input("\nEnter your hardware/market inquiry (or type 'exit' to quit): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nShutting down engine. Goodbye!")
                break
                
            # Initialize our clean graph state data layout
            initial_state: AgentState = {
                "user_query": user_input,
                "sub_questions": [],
                "research_data": [],
                "vetted_facts": [],
                "final_report": ""
            }
            
            print("\n Multi-Agent Pipeline Initiated. Processing inputs...")
            
            # Stream events out of the LangGraph runtime engine in real-time
            final_computed_state = None
            for event in agent_app.stream(initial_state):
                # The event dictionary matches node names to their state mutations
                for node_name, state_update in event.items():
                    print(f"\n[GRAPH LOG]  Node '{node_name}' finished execution.")
                    # Keep track of the accumulating state object
                    final_computed_state = state_update
            
            # Once the waterfall cycle reaches END, extract and display the final report
            print("\n==================================================================")
            print(" FINAL GENERATED ANALYSIS REPORT")
            print("==================================================================")
            
            # Safely fetch report string out of the stream's final collected payload dictionary
            # LangGraph combines node updates, so we check our target key
            if final_computed_state and "final_report" in final_computed_state:
                print(final_computed_state["final_report"])
            else:
                # Fallback extraction directly from state check if needed
                print("Error: Final report was not written to state successfully.")
                
            print("\n==================================================================")
            
        except Exception as e:
            print(f"\nA critical runtime error occurred in the orchestrator: {e}")

if __name__ == "__main__":
    main()