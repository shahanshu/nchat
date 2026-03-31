from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun 


from langchain_core.prompts import ChatPromptTemplate  


from chains import retriever, retrieval_grader, rag_chain, llm  
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
    
    # 1. Perform the raw search (this gets the ugly text)
    raw_web_results = web_search_tool.invoke(question)
    
    # 2. Create a prompt to tell the LLM to clean up and format the answer
    web_search_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. You just searched the web to answer the user's question because it was outside their course syllabus. 

INSTRUCTIONS:
1. Read the provided Web Search Results and synthesize a clear, accurate, and easy-to-read answer.
2. Format your answer beautifully using Markdown (use headings, bullet points, and bold text).
3. Start your response with an italicized note: *"I searched the web for this since it is outside your syllabus context."*
4. Ignore any weird spacing or spelling errors in the raw web text. Make your final output perfect.
5. If the web results do not contain the answer, politely say you couldn't find a good answer online."""),
        ("human", "User Question: {question}\n\nWeb Search Results:\n{web_results}")
    ])
    
    # 3. Pass the ugly results to the LLM to generate a beautiful markdown answer
    web_chain = web_search_prompt | llm
    response = web_chain.invoke({"question": question, "web_results": raw_web_results})
    
    # 4. Return the beautifully formatted text
    return {"generation": response.content, "question": question}
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