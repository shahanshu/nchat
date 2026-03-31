import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ==========================================
# 1. SETUP EMBEDDINGS & VECTORSTORE
# ==========================================
# This vectorstore is used by graph.py for filtered retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ==========================================
# 2. SETUP LLM (Llama 3.3 70B for High Accuracy)
# ==========================================

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# ==========================================
# 3. THE GRADER (Checks if document is relevant)
# ==========================================
system_grader = """You are an intelligent syllabus grader. 
Assess if the retrieved document contains information relevant to the student's question.

RULES:
1. GREETINGS: If the question is a greeting (hi, hello, etc.), output 'yes'.
2. SUBJECT MATTER: If the document contains keywords related to the student's question (even if it's just a chapter title or a marks list), output 'yes'.
3. SYLLABUS: If the user asks about marks or weightage and the document is a syllabus list, output 'yes'.

Output exactly 'yes' or 'no'."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n Student question: {question}")
])

retrieval_grader = grade_prompt | llm | StrOutputParser()

# ==========================================
# 4. THE GENERATOR (The Main Brain)
# ==========================================
system_generator = """You are a highly accurate AI Teaching Assistant for IOE Engineering students. 
You provide information about subjects like Artificial Intelligence (AI), Operating Systems (OS), and Database Management Systems (DBMS).

The context you receive is labeled with two specific tags:
- [SYLLABUS DATA]: This is the official list of Units, Chapters, and Marks. 
- [COURSE CONTENT]: This is detailed technical knowledge from textbooks/PDFs.

STRICT OPERATING RULES:
1. MARKS & CHAPTERS: If the user asks about marks, weightage, or "what is in chapter X", you MUST prioritize the information in [SYLLABUS DATA]. 
2. ACCURACY: Do not invent or guess marks. If [SYLLABUS DATA] says marks are "included" or "Total: 80", report exactly that.
3. TECHNICAL QUESTIONS: For questions like "Explain Paging" or "What is A* search?", use the [COURSE CONTENT] to provide a detailed technical answer.
4. AMBIGUITY: If the user asks a general question (e.g., "What are the marks for Unit 1?"), check if multiple subjects are in the context. If so, ask: "Are you asking about AI, OS, or DBMS?"
5. FORMATTING: Always use Markdown. Use bold headers for Unit names and bullet points for lists.

Context:
{context}"""

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_generator),
    ("human", "{question}")
])

# This chain is called by generate_node in graph.py
rag_chain = generate_prompt | llm