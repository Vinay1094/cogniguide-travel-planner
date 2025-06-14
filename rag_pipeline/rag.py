import os
import json
import numpy as np
import faiss
import google.generativeai as genai

print("RAG: Starting rag.py script...")

# Configure Google API key for embeddings
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("RAG: Google Generative AI configured successfully.")
except Exception as e:
    print(f"RAG ERROR: Failed to configure Google Generative AI: {e}")
    # It's critical to exit if API key is not set, as nothing else will work
    import sys
    sys.exit(1) # Exit the process if API key is not set
print("\n--- Listing available models that support generateContent ---")
found_model = False
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  Found model: {m.name}")
            found_model = True
except Exception as e:
    print(f"  Error listing models: {e}")
if not found_model:
    print("  No models found that support 'generateContent' with this API key/region.")
print("---------------------------------------------------------\n")


# Function to generate embeddings using Google's API
def get_embedding(text_chunk):
    try:
        print(f"RAG: Generating embedding for chunk of length {len(text_chunk)}...")
        response = genai.embed_content(model="models/embedding-001", content=text_chunk)
        if "embedding" not in response:
            raise ValueError(f"Embedding response missing 'embedding' key: {response}")
        print("RAG: Embedding generated successfully.")
        return response["embedding"]
    except Exception as e:
        print(f"RAG ERROR: Failed to generate embedding: {e}")
        return None

class RAGPipeline:
    def __init__(self, index_path='rag_pipeline/faiss_index.bin', metadata_path='rag_pipeline/chunks_metadata.json'):
        print(f"RAG: Initializing RAGPipeline. Index path: {index_path}, Metadata path: {metadata_path}")
        try:
            self.index = faiss.read_index(index_path)
            print("RAG: FAISS index loaded successfully.")
        except Exception as e:
            print(f"RAG ERROR: Failed to load FAISS index from {index_path}: {e}")
            raise # Re-raise to crash early and show the error

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.chunks_metadata = json.load(f)
            print("RAG: Chunks metadata loaded successfully.")
        except Exception as e:
            print(f"RAG ERROR: Failed to load chunks metadata from {metadata_path}: {e}")
            raise # Re-raise to crash early and show the error

        print("RAG: RAGPipeline initialized.")

    def retrieve_chunks(self, query, k=3):
        print(f"RAG: Retrieving chunks for query: '{query}'")
        query_embedding = get_embedding(query)

        if query_embedding is None:
            print("RAG: Query embedding failed, returning empty chunks.")
            return []

        query_embedding = np.array([query_embedding], dtype=np.float32)

        try:
            D, I = self.index.search(query_embedding, k)
            print(f"RAG: FAISS search completed. Distances: {D}, Indices: {I}")
        except Exception as e:
            print(f"RAG ERROR: FAISS search failed: {e}")
            return [] # Return empty if search fails

        retrieved_chunks = []
        for i in I[0]:
            if i != -1 and i < len(self.chunks_metadata): # Ensure index is valid and within bounds
                retrieved_chunks.append(self.chunks_metadata[i])
            else:
                print(f"RAG WARNING: Invalid index {i} found during chunk retrieval.")
        print(f"RAG: Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    def get_answer(self, query):
        print(f"RAG: Getting answer for query: '{query}'")
        try:
            retrieved_chunks = self.retrieve_chunks(query)

            if not retrieved_chunks:
                print("RAG: No relevant chunks found.")
                return "No relevant information found for your query in the documents."

            context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
            print(f"RAG: Context prepared for Gemini (length: {len(context)}).")

            model = genai.GenerativeModel('models/gemini-1.5-pro')
            prompt = f"""
            You are a helpful travel assistant. Based on the following context, answer the user's query.
            If the context does not contain enough information to answer the question, state that you cannot answer from the provided context.

            Context:
            {context}

            User Query: {query}

            Answer:
            """
            response = model.generate_content(prompt)
            print("RAG: Gemini content generated successfully.")
            return response.text
        except Exception as e:
            print(f"RAG ERROR: Error in get_answer method: {e}")
            return f"An error occurred while generating the answer: {str(e)}"