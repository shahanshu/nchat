import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pinecone import Pinecone

# 1. Load keys
load_dotenv()

# 2. Test Groq (LLM)
try:
    print("🤖 Testing Groq Connection...")
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile" # or "mixtral-8x7b-32768"
    )
    response = llm.invoke("Hello, are you ready to help engineering students?")
    print(f"✅ Groq Response: {response.content}")
except Exception as e:
    print(f"❌ Groq Error: {e}")

# 3. Test Pinecone (Vector DB)
try:
    print("\n🌲 Testing Pinecone Connection...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # List current indexes to confirm connection
    indexes = pc.list_indexes()
    print(f"✅ Pinecone Indexes found: {[i.name for i in indexes]}")
    
    # Check if your specific index exists, if not, print a warning
    target_index = os.environ.get("PINECONE_INDEX_NAME")
    if not any(i.name == target_index for i in indexes):
        print(f"⚠️ Warning: Index '{target_index}' not found. We will create it in the next step.")
    else:
        print(f"✅ Index '{target_index}' is ready.")

except Exception as e:
    print(f"❌ Pinecone Error: {e}")