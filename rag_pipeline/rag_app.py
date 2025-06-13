import os
import json
import numpy as np
import os
import json
import numpy as np
import faiss
import google.generativeai as genai # Add this line

# Configure Google API key for embeddings
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY')) # Add this line, assuming GOOGLE_API_KEY is an env var

# --- Configuration ---
# Define paths relative to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, 'faiss_index.bin')
CHUNKS_METADATA_PATH = os.path.join(SCRIPT_DIR, 'chunks_metadata.json')
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Same model used for creating embeddings
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest' # The Gemini model you confirmed is working

# --- Load Components ---
# Function to generate embeddings using Google's API
def get_embedding(text_chunk):
    try:
        # Use Google's embedding model
        # 'models/embedding-001' is a common embedding model from Google
        response = genai.embed_content(model="models/embedding-001", content=text_chunk)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Modify your RAGPipeline class to use this new get_embedding function
class RAGPipeline:
    def __init__(self, index_path='rag_pipeline/faiss_index.bin', metadata_path='rag_pipeline/chunks_metadata.json'):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)
        # No SentenceTransformer model needs to be loaded here anymore
        # self.model = SentenceTransformer('all-MiniLM-L6-v2') # Remove or comment out this line

    def retrieve_chunks(self, query, k=3):
        query_embedding = get_embedding(query) # Use the new get_embedding function
        if query_embedding is None:
            return []
        # FAISS expects numpy array of float32, reshaped to (1, embedding_dim)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Perform similarity search
        D, I = self.index.search(query_embedding, k)

        retrieved_chunks = []
        for i in I[0]:
            if i != -1: # Ensure the index is valid
                retrieved_chunks.append(self.chunks_metadata[i])
        return retrieved_chunks

    # Your get_answer method (or similar) will use retrieve_chunks, which now uses Google Embeddings
    # Example (ensure your get_answer calls retrieve_chunks internally):
    # def get_answer(self, user_query):
    #     relevant_chunks = self.retrieve_chunks(user_query)
    #     # ... rest of your logic ...
    #     return final_answer

# Load the FAISS index
print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    print("Ensure 'faiss_index.bin' exists in the 'rag_pipeline' directory.")
    exit()

# Load the chunks metadata (original text)
print(f"Loading chunks metadata from {CHUNKS_METADATA_PATH}...")
try:
    with open(CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
        chunks_metadata = json.load(f)
    print(f"Loaded {len(chunks_metadata)} chunks metadata.")
except Exception as e:
    print(f"Error loading chunks metadata: {e}")
    print("Ensure 'chunks_metadata.json' exists in the 'rag_pipeline' directory.")
    exit()

# Configure the Gemini API key
print("Configuring Gemini API key...")
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Gemini LLM '{GEMINI_MODEL_NAME}' configured.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it before running this script.")
    exit()
except Exception as e:
    print(f"Error configuring Gemini LLM or API: {e}")
    exit()

# --- RAG Function ---
def get_rag_response(query: str, top_k: int = 5) -> str:
    print(f"\nProcessing query: '{query}'")

    # 1. Embed the user query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # 2. Perform similarity search in FAISS
    # D: distances, I: indices of the top_k nearest neighbors
    D, I = index.search(query_embedding, top_k)
    
    retrieved_chunk_indices = I[0] # Get the indices for the first query
    print(f"Retrieved top {top_k} chunk indices: {retrieved_chunk_indices}")

    # 3. Retrieve the full text of the chunks
    retrieved_contexts = []
    for idx in retrieved_chunk_indices:
        if 0 <= idx < len(chunks_metadata):
            retrieved_contexts.append(chunks_metadata[idx]['text'])
        else:
            print(f"Warning: Chunk index {idx} out of bounds.")

    if not retrieved_contexts:
        return "Sorry, I couldn't find any relevant information in the knowledge base."

    # 4. Construct the augmented prompt for the LLM
    context_str = "\n".join([f"CONTEXT:\n{c}" for c in retrieved_contexts])
    
    # You can experiment with different prompt formats
    augmented_prompt = f"""You are CogniGuide, an AI assistant specializing in travel information.
Use the following context to answer the user's question. If the information is not in the context, state that you don't have enough information.
Keep your answer concise and directly related to the provided context.

{context_str}

User's Question: {query}
CogniGuide's Answer:
"""
    print("\n--- Sending Augmented Prompt to LLM ---")
    # print(augmented_prompt) # Uncomment this to see the full prompt sent to LLM
    print("---------------------------------------")

    # 5. Send the augmented prompt to Gemini LLM
    try:
        response = gemini_llm.generate_content(augmented_prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content with LLM: {e}")
        return "An error occurred while generating a response from the AI. Please try again."

# --- Main Application Loop ---
if __name__ == "__main__":
    print("\n--- CogniGuide RAG System Ready ---")
    print("Type your travel questions. Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            print("Exiting CogniGuide. Goodbye!")
            break
        
        response_text = get_rag_response(user_query)
        print("\nCogniGuide's Response:")
        print(response_text)
        print("\n----------------------------------")