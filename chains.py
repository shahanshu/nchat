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
print("Loading Multi-Subject Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# INCREASED k=6: We pull more chunks now because we have multiple subjects in the DB
retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) 

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
2. RELEVANT: If the input is about engineering subjects (like AI, OS, Embedded Systems, etc.), syllabus structure, topics, marks, exams, or computer science concepts, AND the document contains related info, grade it as 'yes'.
3. IRRELEVANT / OUT OF SYLLABUS: If the user asks about a completely unrelated topic (e.g., history, sports, cooking, politics) or a technical question with zero relevance to the retrieved documents, grade it as 'no'.

Output exactly 'yes' or 'no' without any extra words or punctuation."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n Student question: {question}")
])

retrieval_grader = grade_prompt | llm | StrOutputParser()

# ==========================================
# 4. THE GENERATOR (Answers the question)
# ==========================================
system_generator = """You are a friendly and helpful AI Teaching Assistant for Engineering (IOE) students. You assist with multiple subjects like Artificial Intelligence (AI), Operating Systems (OS), Embedded Systems, etc.

INSTRUCTIONS:
1. CHITCHAT: If the user says hello, asks how you are, or says thanks, respond politely and naturally.
2. IDENTIFY THE SUBJECT: Pay close attention to the [SUBJECT: ...] and [TYPE: ...] tags at the beginning of the context blocks below. Ensure your answer matches the specific subject the user is asking about. 
3. AMBIGUITY: If the user asks a general question (e.g., "What is chapter 1?" or "What are the marks for unit 2?") without specifying a subject, politely ask them to clarify which subject they mean (e.g., AI, OS, or Embedded).
4. SYLLABUS & MARKS: For questions about marks allocations, weightage, topics, or course structure, heavily base your answer on context tagged with [TYPE: SYLLABUS & MARKS].
5. COURSE CONTENT: For conceptual questions (e.g., "What is A* search?"), base your answer on context tagged with [TYPE: COURSE CONTENT].
6. OUT OF CONTEXT: If a technical question isn't explicitly in the context, provide a brief, accurate answer based on your general knowledge, but add a note stating: "(Note: This specific detail might not be explicitly in your syllabus documents)."
7. Keep your answers clear, encouraging, and format them nicely with bullet points or bold text where appropriate.

Syllabus/Content Context:
{context}"""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_generator),
    ("human", "{question}")
])

rag_chain = generate_prompt | llm