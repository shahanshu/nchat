import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
PDF_DIR = "./data/pdfs"
MD_DIR = "./data/mds"
VECTOR_DB_DIR = "./chroma_db"

def ingest_all_subjects():
    # 0. Clean up old database to avoid duplicate/old data
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Cleaning up old database at {VECTOR_DB_DIR}...")
        shutil.rmtree(VECTOR_DB_DIR)

    all_chunks = []
    
    # 1. Setup Markdown Header Splitter for Syllabus
    # This keeps all Units of a subject together in one chunk
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    print("1. Processing Markdown Files (Syllabus & Marks)...")
    md_files = glob.glob(os.path.join(MD_DIR, "*.md"))
    for file_path in md_files:
        subject_name = os.path.basename(file_path).replace(".md", "").upper()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # This will now keep all Units of AI together in one chunk
        md_docs = md_splitter.split_text(content)
        
        for doc in md_docs:
            doc.metadata["subject"] = subject_name
            doc.metadata["source_type"] = "syllabus" # CRITICAL for prioritized retrieval
            # Add a prefix to the content so the LLM knows this is the source of truth for marks
            doc.page_content = f"[SYLLABUS DATA - SUBJECT: {subject_name}]\n" + doc.page_content
        
        all_chunks.extend(md_docs)
        print(f"   -> Processed {subject_name} Syllabus.")

    # 2. Setup Recursive Splitter for PDF Course Content
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    print("\n2. Processing PDF Files (Detailed Course Content)...")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    for file_path in pdf_files:
        subject_name = os.path.basename(file_path).replace(".pdf", "").upper()
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split PDFs into smaller pieces
        pdf_chunks = pdf_splitter.split_documents(docs)
        
        for chunk in pdf_chunks:
            chunk.metadata["subject"] = subject_name
            chunk.metadata["source_type"] = "content" # CRITICAL for prioritized retrieval
            chunk.page_content = f"[COURSE CONTENT - SUBJECT: {subject_name}]\n" + chunk.page_content
        
        all_chunks.extend(pdf_chunks)
        print(f"   -> Processed {subject_name} PDF into {len(pdf_chunks)} chunks.")

    if not all_chunks:
        print("\nError: No documents were processed. Check your data folders.")
        return

    print(f"\n3. Saving {len(all_chunks)} total chunks to Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"\n✅ Success! New database saved at '{VECTOR_DB_DIR}'.")

if __name__ == "__main__":
    ingest_all_subjects()