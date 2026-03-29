from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver # <-- NEW: For pausing the graph
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun # <-- NEW: For web search

from chains import retriever, retrieval_grader, rag_chain

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str 
    documents: List[Document]

# Initialize Web Search Tool
web_search_tool = DuckDuckGoSearchRun()

def retrieve_node(state: GraphState):
    print("\n--- NODE: RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents_node(state: GraphState):
    print("\n--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs =[]
    web_search = "No"
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.strip().lower()
        
        if "yes" in grade:
            print("   -> GRADE: RELEVANT")
            filtered_docs.append(d)
        else:
            print("   -> GRADE: IRRELEVANT")
    
    if len(filtered_docs) == 0:
        print("   -> ALL DOCUMENTS IRRELEVANT: FLAG FOR WEB SEARCH!")
        web_search = "Yes"
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate_node(state: GraphState):
    print("\n--- NODE: GENERATE ANSWER (FROM SYLLABUS) ---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation.content}

def web_search_node(state: GraphState):
    print("\n--- NODE: EXECUTING WEB SEARCH ---")
    question = state["question"]
    
    # Perform the search
    docs = web_search_tool.invoke(question)
    
    # Format the web results
    msg = f"According to the web:\n{docs}"
    return {"generation": msg, "question": question}

def decide_to_generate(state: GraphState):
    if state["web_search"] == "Yes":
        return "websearch"
    else:
        return "generate"

# ==========================================
# BUILD THE GRAPH
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("websearch", web_search_node) # <-- Updated to real web search

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate"},
)

workflow.add_edge("generate", END)
workflow.add_edge("websearch", END)

# Set up Memory to pause the graph
memory = MemorySaver()
# Compile with an interrupt (pause) right before the websearch node runs
app = workflow.compile(checkpointer=memory, interrupt_before=["websearch"])

# ==========================================
# RUN AND TEST
# ==========================================
if __name__ == "__main__":
    # We need a "thread" to remember the conversation state
    config = {"configurable": {"thread_id": "1"}}

    print("\n=== ASKING AN OUT-OF-SYLLABUS QUESTION ===")
    question = "What is the capital of France?"
    
    # Run the graph until it finishes OR hits a pause (interrupt)
    for output in app.stream({"question": question}, config):
        pass 
    
    # Check the current state of the graph
    current_state = app.get_state(config)
    
    # If the graph is paused, it will have a "next" node pending
    if current_state.next and current_state.next[0] == "websearch":
        print("\n[SYSTEM]: 🛑 The chatbot has paused execution.")
        user_input = input("This topic is outside the syllabus. Would you like me to search the web? (Y/N): ")
        
        if user_input.strip().lower() == 'y':
            print("\nResuming graph and searching the web...")
            # Passing `None` tells the graph to just resume from where it paused
            for output in app.stream(None, config): 
                pass
            
            final_state = app.get_state(config)
            print(f"\nFINAL ANSWER:\n{final_state.values['generation']}")
        else:
            print("\nFINAL ANSWER:\nOkay, I have cancelled the search. Please ask a syllabus-related question!")