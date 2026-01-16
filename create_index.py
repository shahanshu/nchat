import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "ioe-syllabus")

# 2. Check if index exists
existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}...")
    try:
        pc.create_index(
            name=index_name,
            dimension=384, # We will use 'all-MiniLM-L6-v2' (free, fast, good for English)
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Free tier is usually here
            )
        )
        print("Index creating... waiting for initialization.")
        time.sleep(10) # Wait for it to spin up
        print("✅ Index successfully created!")
    except Exception as e:
        print(f"❌ Error creating index: {e}")
else:
    print(f"✅ Index '{index_name}' already exists.")