import streamlit as st
import langchain_community, langchain_core
from langchain_core.prompts import PromptTemplate 
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# ==============================================================================
# 0. CONFIGURATION & INITIALIZATION (Cache the heavy lifting)
# ==============================================================================

# NOTE: Replace 'your_vector_store_path' with the actual path where you saved your FAISS index
VECTOR_STORE_PATH = "faiss_index",

# Define the Prompt Template (Assuming you defined this earlier)
SYSTEM_TEMPLATE = """
You are a helpful and friendly assistant. 
Use the following context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

QUESTION: {question}
"""

@st.cache_resource
def get_rag_chain():
    """Initializes and returns the RAG chain."""
    try:
        # Load the Embeddings Model
        # Use HuggingFaceEmbeddings from the new package
        embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
        
        # Load the Vector Store
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH, 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True # Necessary for loading
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize the LLM (Ensure Ollama is running locally!)
        llm = Ollama(model="gemma3:1b", temperature=0.1)

        # Create the PromptTemplate
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=SYSTEM_TEMPLATE,
        )

        # Build the ConversationalRetrievalChain (using the corrected method)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        
        return chain
    
    except Exception as e:
        st.error(f"Error loading RAG components. Is Ollama running and is '{VECTOR_STORE_PATH}' correct? Error: {e}")
        st.stop()
        
# Initialize the chain
chain = get_rag_chain()

# ==============================================================================
# 1. STREAMLIT UI SETUP
# ==============================================================================

st.set_page_config(page_title="Prasad's Local RAG Chatbot Demo", layout="wide")
st.title("Local Gemma RAG Chatbot (Ollama)")
st.caption("Ask me about the documents in the vector store.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your documents today?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# 2. USER INPUT AND RAG CHAIN CALL
# ==============================================================================

if prompt := st.chat_input("Your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Extract previous chat history for the chain
            # The chain expects a list of tuples: [(human_msg, ai_msg), ...]
            chat_history = []
            for i in range(1, len(st.session_state.messages)):
                if st.session_state.messages[i]["role"] == "user":
                    user_msg = st.session_state.messages[i]["content"]
                    # Find the corresponding AI response (it should be the next message)
                    if i + 1 < len(st.session_state.messages) and st.session_state.messages[i + 1]["role"] == "assistant":
                        ai_msg = st.session_state.messages[i + 1]["content"]
                        chat_history.append((user_msg, ai_msg))

            # Call the ConversationalRetrievalChain
            result = chain.invoke(
                {"question": prompt, "chat_history": chat_history}
            )

            response = result["answer"]
            source_docs = result["source_documents"]

            # Display the main answer
            st.markdown(response)

            # Optionally, display the source documents
            with st.expander("Show Sources"):
                for i, doc in enumerate(source_docs):
                    st.write(f"**Source {i+1}**: {doc.metadata.get('source', 'N/A')}")
                    st.caption(doc.page_content[:200] + "...") # Show first 200 chars

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})