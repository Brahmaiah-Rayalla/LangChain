"""
LangGraph Example - Building stateful applications with graphs.

Note: LangGraph requires langgraph package.
Install with: pip install langgraph
"""
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common import print_section, print_result, OPENAI_API_KEY


def basic_graph_example():
    """Example of a basic LangGraph state graph."""
    print_section("LangGraph Basic Graph Example")

    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Annotated
        import operator

        # Define state
        class GraphState(TypedDict):
            messages: Annotated[list, operator.add]

        # Define nodes
        def agent_node(state: GraphState):
            return {"messages": ["Agent: Processing request"]}

        def tool_node(state: GraphState):
            return {"messages": ["Tool: Executing action"]}

        # Build graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool", tool_node)

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", "tool")
        workflow.add_edge("tool", END)

        # Compile graph
        app = workflow.compile()

        print_result("Graph Created", "Agent -> Tool -> END")

        # Run graph
        result = app.invoke({"messages": ["User: Hello"]})
        print_result("Graph Execution", result)

    except ImportError:
        print("⚠️  langgraph not installed. Install it with: pip install langgraph")
    except Exception as e:
        print(f"⚠️  Error: {e}")


def conditional_graph_example():
    """Example of a graph with conditional edges."""
    print_section("LangGraph Conditional Graph Example")

    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict, Annotated, Literal
        import operator

        # Define state
        class GraphState(TypedDict):
            messages: Annotated[list, operator.add]
            should_continue: bool

        # Define nodes
        def agent_node(state: GraphState):
            state["messages"].append("Agent: Analyzing")
            return state

        def tool_node(state: GraphState):
            state["messages"].append("Tool: Executing")
            return state

        def end_node(state: GraphState):
            state["messages"].append("END: Complete")
            return state

        # Define routing logic
        def should_continue(state: GraphState) -> Literal["tool", "end"]:
            if state.get("should_continue", False):
                return "tool"
            return "end"

        # Build graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tool", tool_node)
        workflow.add_node("end", end_node)

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tool": "tool",
                "end": "end"
            }
        )
        workflow.add_edge("tool", "agent")
        workflow.add_edge("end", END)

        # Compile graph
        app = workflow.compile()

        print_result("Graph Created", "Agent with conditional routing")

        # Run graph
        result = app.invoke({"messages": [], "should_continue": True})
        print_result("Graph Execution", result)

    except ImportError:
        print("⚠️  langgraph not installed. Install it with: pip install langgraph")
    except Exception as e:
        print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    basic_graph_example()
    conditional_graph_example()
