import streamlit as st
import os
import tempfile
import time

# Set page configuration
st.set_page_config(
    page_title="Smart QA System",
    page_icon="🧠",
    layout="wide"
)

# Hide Streamlit menu and footer for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS for UI improvement
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Import backend logic
# Importing will load the models into memory (Streamlit keeps this memory active)
with st.spinner("Loading AI Models... Please wait (this can take a moment)..."):
    from smart_qa_complete import add_material, ask_question, documents

st.title("🧠 Smart Document & Video Assistant")
st.markdown("Ask questions about your uploaded PDFs, Texts, and Educational Videos.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.markdown("Upload files to analyze them.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md", "csv", "mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner(f"Ingesting {uploaded_file.name}... (This may take a while for videos)"):
                try:
                    # Save to temp file but preserve the original filename 
                    # so the system displays "cryptovid" instead of "tmpxyz123"
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                    
                    # Call the backend function
                    success = add_material(temp_path)
                    
                    if success:
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    else:
                        st.error(f"Failed to extract content from {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    st.divider()
    st.header("📚 Loaded Materials")
    if not documents:
        st.info("No documents loaded yet.")
    else:
        for doc_name, chunks in documents.items():
            doc_type = chunks[0]['type'] if chunks else '?'
            st.text(f"✅ {doc_name} ({doc_type}: {len(chunks)} chunks)")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not documents:
         with st.chat_message("assistant"):
             st.warning("Please upload and process a document in the sidebar first!")
             st.session_state.messages.append({"role": "assistant", "content": "Please upload and process a document in the sidebar first!"})
    else:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                answer = ask_question(prompt)
                
            if answer:
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = "Sorry, I couldn't generate an answer. Make sure Ollama is running."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
