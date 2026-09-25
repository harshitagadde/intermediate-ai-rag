from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    route: str
    search_results: List[str]
    response: str

def router_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    if any(k in query for k in ["code", "bug", "python", "error", "api"]):
        state["route"] = "coder"
    elif any(k in query for k in ["news", "search", "latest", "price", "who is"]):
        state["route"] = "web_search"
    else:
        state["route"] = "general"
    return state

def web_search_agent(state: AgentState) -> AgentState:
    # Simulated Tool Calling (Replace with real Tavily API client in production)
    query = state["query"]
    simulated_fetch = [f"Fetched real-time search context for query: '{query}'"]
    state["search_results"] = simulated_fetch
    state["response"] = f"[Search Agent]: Grounded response using live web data -> {simulated_fetch}"
    return state

def coding_agent(state: AgentState) -> AgentState:
    state["response"] = f"[Coding Agent]: Resolved technical problem: '{state['query']}'"
    return state

def general_agent(state: AgentState) -> AgentState:
    state["response"] = f"[General Agent]: Handled standard prompt: '{state['query']}'"
    return state

def build_advanced_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("coder", coding_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("general", general_agent)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"coder": "coder", "web_search": "web_search", "general": "general"}
    )
    workflow.add_edge("coder", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("general", END)

    return workflow.compile()

if __name__ == "__main__":
    app = build_advanced_graph()
    queries = [
        "Fix a python bug in my FastAPI app",
        "What is the latest stock price for NVIDIA?"
    ]
    for q in queries:
        res = app.invoke({"query": q, "route": "", "search_results": [], "response": ""})
        print(f"\nQuery: {q}")
        print(f"Result: {res['response']}")
