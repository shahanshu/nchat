import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
system_grader = """You are an intelligent syllabus grader assessing the relevance of a retrieved document to a user's question.

RULES:
1. CHITCHAT/GREETINGS: If the user's input is a simple greeting or casual conversation (e.g., "hi", "hello", "how are you", "thank you", "who are you"), ALWAYS grade it as 'yes' so it passes through to the assistant.
2. RELEVANT: If the input is about DBMS, syllabus structure, topics, marks, or computer science, AND the document contains related info, grade it as 'yes'.
3. IRRELEVANT / OUT OF SYLLABUS: If the user asks about a completely unrelated topic (e.g., history, sports, current events) or a technical question with zero relevance to the retrieved document, grade it as 'no'.

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
system_generator = """You are a friendly and helpful AI Teaching Assistant for a DBMS (Database Management Systems) course.

INSTRUCTIONS:
1. CHITCHAT: If the user says hello, asks how you are, or says thanks, respond politely and naturally. You do not need to use the syllabus context for this.
2. SYLLABUS QUESTIONS: For questions about DBMS topics, marks allocation, or course structure, base your answer heavily on the provided syllabus context below. Pay close attention to marks allocation and topic distribution.
3. SIMPLE/GENERAL CS QUESTIONS: If the user asks a simple computer science/DBMS question that isn't explicitly in the context, provide a brief, accurate answer based on your knowledge, but add a quick note stating: "(Note: This specific detail might not be in your syllabus)."
4. Keep your answers clear, encouraging, and concise.

Syllabus Context:
{context}"""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_generator),
    ("human", "{question}")
])

rag_chain = generate_prompt | llm