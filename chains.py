import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # <-- NEW

load_dotenv()

# ==========================================
# 1. SETUP RETRIEVER
# ==========================================
print("Loading Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# ==========================================
# 2. SETUP LLM
# ==========================================
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ==========================================
# 3. THE GRADER (Checks if in syllabus)
# ==========================================
# SIMPLIFIED: We just ask the model to output a string now.
system_grader = """You are a strict syllabus grader assessing relevance of a retrieved document to a student's question. 
If the document contains information, topics, or marks allocation related to the question, grade it as 'yes'.
If the document does not contain the answer, or the topic is out of the syllabus context provided, grade it as 'no'.
Output exactly 'yes' or 'no' without any extra words or punctuation."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n Student question: {question}")
])

# Use StrOutputParser to easily get the text output
retrieval_grader = grade_prompt | llm | StrOutputParser()

# ==========================================
# 4. THE GENERATOR (Answers the question)
# ==========================================
system_generator = """You are a helpful, strict teaching assistant. Use the following pieces of retrieved syllabus context to answer the question. 
Pay close attention to marks allocation and topic distribution if the student asks about them.
If the answer is not in the context, you must politely state that the topic is out of the syllabus.
Keep the answer clear, concise, and strictly based on the provided context.

Context: {context}"""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_generator),
    ("human", "{question}")
])

rag_chain = generate_prompt | llm