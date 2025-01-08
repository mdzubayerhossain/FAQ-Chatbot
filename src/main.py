import os
import json
import streamlit as st
from groq import Groq
from typing import List, Dict
import re
from jinachat import embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page configuration
st.set_page_config(
    page_title="Sukhi-Poribar FAQ Bot",
    page_icon="📚",
    layout="centered"
)

class EmbeddingRetriever:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
        
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text using Jina."""
        return embedding.embed(text)
    
    def add_chunk(self, chunk: str):
        """Add a chunk and its embedding to the retriever."""
        if chunk.strip():
            self.chunks.append(chunk)
            self.embeddings.append(self.compute_embedding(chunk))
            
    def find_relevant_chunks(self, query: str, top_k: int = 2) -> List[str]:
        """Find most relevant chunks using cosine similarity."""
        query_embedding = self.compute_embedding(query)
        
        # Convert embeddings to numpy array for batch computation
        embeddings_array = np.array(self.embeddings)
        
        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], embeddings_array)[0]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_indices]

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'([।\n]|\.\s)', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Rough estimate of tokens (characters / 3 for Bengali)
        sentence_length = len(sentence) // 3
        
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Initialize Groq client
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

# Initialize retriever and load FAQ content
@st.cache_resource
def initialize_retriever():
    retriever = EmbeddingRetriever()
    try:
        with open(r"D:\Coding\Llama chatbot\FAQ.txt", encoding='utf-8') as file:
            content = file.read()
        chunks = chunk_text(content)
        for chunk in chunks:
            retriever.add_chunk(chunk)
        return retriever
    except FileNotFoundError:
        st.error("FAQ.txt file not found. Please ensure it exists in the working directory.")
        return None

retriever = initialize_retriever()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def create_system_prompt(query: str) -> str:
    """Create a system prompt using relevant FAQ chunks."""
    if retriever:
        relevant_chunks = retriever.find_relevant_chunks(query)
        relevant_content = "\n\n".join(relevant_chunks)
    else:
        relevant_content = "FAQ content not available."
        
    return f"""You are a helpful assistant specialized in answering questions about Bengali family health. 
    Answer based on these relevant FAQ sections:

    {relevant_content}

    Guidelines:
    1. Answer based solely on the provided FAQ content
    2. If the information isn't in the provided sections, say so
    3. Respond in the same language as the user's question (Bengali or English)
    4. Keep responses clear and concise
    5. Stay focused on the specific question asked"""

# UI Elements
st.title("📚 চ্যাটবট- এ আপনাকে স্বাগতম")
st.markdown("পরিবার পরিকল্পনা বিষয়ক তথ্য জানতে প্রশ্ন করুন।")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input("আপনার প্রশ্ন লিখুন... / Type your question...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Create system prompt with relevant content
    system_prompt = create_system_prompt(user_prompt)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Get response from LLM
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### About This Chatbot")
    st.markdown("""
    This chatbot is designed to answer questions about family health based on the সুখী পরিবার (Sukhi Poribar) FAQ.
    
    You can:
    - Ask questions in Bengali or English
    - Get information about family health topics
    - Receive answers based on the official FAQ content
    """)
    
    # Add debug information in sidebar if needed
    if st.checkbox("Show Debug Info"):
        if retriever:
            st.write(f"Number of FAQ chunks: {len(retriever.chunks)}")
        if user_prompt:
            st.write("Relevant content length:", len(create_system_prompt(user_prompt)))    