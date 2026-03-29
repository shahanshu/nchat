from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun 

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
            print("   -> GRADE: RELEVANT (or Chitchat)")
            filtered_docs.append(d)
        else:
            print("   -> GRADE: IRRELEVANT")
    
    # If all documents are irrelevant (and it's not chitchat), trigger web search
    if len(filtered_docs) == 0:
        print("   -> ALL DOCUMENTS IRRELEVANT: FLAG FOR WEB SEARCH!")
        web_search = "Yes"
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate_node(state: GraphState):
    print("\n--- NODE: GENERATE ANSWER ---")
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
    msg = f"**Here is what I found from the web:**\n\n{docs}"
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
workflow.add_node("websearch", web_search_node) 

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate"},
)

workflow.add_edge("generate", END)
workflow.add_edge("websearch", END)

# Set up Memory to pause the graph for user confirmation
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["websearch"])