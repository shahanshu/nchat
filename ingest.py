import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
PDF_DIR = "./data/pdfs"
MD_DIR = "./data/mds"
VECTOR_DB_DIR = "./chroma_db"

def ingest_all_subjects():
    all_documents = []
    
    print("1. Loading Markdown Files (Syllabus & Marks)...")
    md_files = glob.glob(os.path.join(MD_DIR, "*.md"))
    for file_path in md_files:
        # Extract subject name from filename (e.g., "ai.md" -> "AI")
        subject_name = os.path.basename(file_path).replace(".md", "").upper()
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        
        # Add metadata and prepend subject to content so the LLM knows what subject this is
        for doc in docs:
            doc.metadata["subject"] = subject_name
            doc.metadata["doc_type"] = "syllabus_marks"
            # INJECT HEADER: This is the magic trick for multi-subject RAG
            doc.page_content = f"[SUBJECT: {subject_name} | TYPE: SYLLABUS & MARKS]\n\n" + doc.page_content
        
        all_documents.extend(docs)
        print(f"   -> Loaded {subject_name} Markdown.")

    print("\n2. Loading PDF Files (Course Content)...")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    for file_path in pdf_files:
        # Extract subject name from filename (e.g., "ai.pdf" -> "AI")
        subject_name = os.path.basename(file_path).replace(".pdf", "").upper()
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["subject"] = subject_name
            doc.metadata["doc_type"] = "course_content"
            # INJECT HEADER
            doc.page_content = f"[SUBJECT: {subject_name} | TYPE: COURSE CONTENT]\n\n" + doc.page_content
        
        all_documents.extend(docs)
        print(f"   -> Loaded {subject_name} PDF.")

    if not all_documents:
        print("\nError: No files found in ./data/pdfs or ./data/mds")
        return

    print("\n3. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"   -> Split into {len(chunks)} chunks.")

    print("\n4. Saving to Local Chroma Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"\n✅ Success! Multi-subject Vector database saved locally at '{VECTOR_DB_DIR}'.")

if __name__ == "__main__":
    ingest_all_subjects()