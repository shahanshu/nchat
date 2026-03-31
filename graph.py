from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_core.prompts import ChatPromptTemplate  

# Import vectorstore directly to use metadata filtering
from chains import vectorstore, retrieval_grader, rag_chain, llm  

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str 
    documents: List[Document]

# Initialize Web Search Tool
web_search_tool = DuckDuckGoSearchRun()

def retrieve_node(state: GraphState):
    print("\n--- NODE: RETRIEVE (SYLLABUS-FIRST SEARCH) ---")
    question = state["question"]
    
    # Increase k=5 to ensure we get the full syllabus even if it's large
    print("   -> Searching Syllabus Chunks...")
    syllabus_docs = vectorstore.similarity_search(
        question, 
        k=5, 
        filter={"source_type": "syllabus"}
    )
    
    # Search for technical details
    print("   -> Searching Course Content Chunks...")
    content_docs = vectorstore.similarity_search(
        question, 
        k=4, 
        filter={"source_type": "content"}
    )
    
    combined_documents = syllabus_docs + content_docs
    print(f"   -> Retrieved {len(syllabus_docs)} syllabus and {len(content_docs)} content chunks.")
    
    return {"documents": combined_documents, "question": question}

def grade_documents_node(state: GraphState):
    print("\n--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = "No"
    
    for d in documents:
        # Check relevance
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.strip().lower()
        
        if "yes" in grade:
            print("   -> GRADE: RELEVANT")
            filtered_docs.append(d)
        else:
            print("   -> GRADE: IRRELEVANT")
    
    # If even after two searches we have nothing, trigger web search
    if len(filtered_docs) == 0:
        print("   -> NO RELEVANT DOCUMENTS FOUND: FLAG FOR WEB SEARCH!")
        web_search = "Yes"
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate_node(state: GraphState):
    print("\n--- NODE: GENERATE ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    # The rag_chain in chains.py will handle the synthesis
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation.content}

def web_search_node(state: GraphState):
    print("\n--- NODE: EXECUTING WEB SEARCH ---")
    question = state["question"]
    
    raw_web_results = web_search_tool.invoke(question)
    
    web_search_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. You just searched the web because the question was outside the provided syllabus. 

INSTRUCTIONS:
1. Synthesize a clear, accurate, and easy-to-read answer.
2. Format your answer beautifully using Markdown.
3. Start your response with: *"I searched the web for this since it is outside your syllabus context."*"""),
        ("human", "User Question: {question}\n\nWeb Search Results:\n{web_results}")
    ])
    
    web_chain = web_search_prompt | llm
    response = web_chain.invoke({"question": question, "web_results": raw_web_results})
    
    return {"generation": response.content, "question": question}

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

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["websearch"])