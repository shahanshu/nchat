import streamlit as st
import uuid
from graph import app  # Importing the compiled LangGraph from your graph.py

st.set_page_config(page_title="Syllabus AI Assistant", page_icon="📚")

st.title("📚 Syllabus-Aligned Chatbot")
st.markdown("Ask me anything about your **DBMS Syllabus**. If I can't find it, I'll ask to search the web!")

# 1. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "waiting_for_approval" not in st.session_state:
    st.session_state.waiting_for_approval = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 2. Function to run the graph
def run_chatbot(user_input):
    # If we are resuming after approval
    if user_input is None:
        inputs = None
    else:
        inputs = {"question": user_input}

    # Stream the graph execution
    final_output = ""
    for output in app.stream(inputs, config):
        for key, value in output.items():
            if "generation" in value:
                final_output = value["generation"]
    
    # Check if the graph is currently paused at 'websearch'
    snapshot = app.get_state(config)
    if snapshot.next and snapshot.next[0] == "websearch":
        st.session_state.waiting_for_approval = True
        return "⚠️ This topic is outside the syllabus. Would you like me to search the web for an answer?"
    
    return final_output

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if not st.session_state.waiting_for_approval:
    if prompt := st.chat_input("Ask a DBMS question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = run_chatbot(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Force a rerun to show buttons if needed
            if st.session_state.waiting_for_approval:
                st.rerun()

# 5. Handle Web Search Permission (The "Buttons")
if st.session_state.waiting_for_approval:
    st.warning("Confirmation Required")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes, Search Web"):
            with st.chat_message("assistant"):
                with st.spinner("Searching the web..."):
                    response = run_chatbot(None) # Pass None to resume
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.waiting_for_approval = False
            st.rerun()
            
    with col2:
        if st.button("❌ No, Cancel"):
            st.session_state.messages.append({"role": "assistant", "content": "Search cancelled. Please ask a syllabus question."})
            st.session_state.waiting_for_approval = False
            # We must "fake" resume and exit or just clear the state
            # For simplicity in this local app, we just reset the flag
            st.rerun()