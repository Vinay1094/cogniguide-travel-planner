import os
import json
import numpy as np
import faiss
import google.generativeai as genai

# Configure Google API key for embeddings
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

# Function to generate embeddings using Google's API
def get_embedding(text_chunk):
    try:
        # Use Google's embedding model
        response = genai.embed_content(model="models/embedding-001", content=text_chunk)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

class RAGPipeline:
    def __init__(self, index_path='rag_pipeline/faiss_index.bin', metadata_path='rag_pipeline/chunks_metadata.json'):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)
        # Ensure there is NO self.model = SentenceTransformer(...) line here anymore.

    def retrieve_chunks(self, query, k=3):
        # The query_embedding should come directly from the standalone get_embedding function
        query_embedding = get_embedding(query)

        if query_embedding is None:
            # Handle cases where embedding failed (e.g., API key issue, network)
            return []

        # Ensure the embedding is a numpy array of float32 for FAISS
        query_embedding = np.array([query_embedding], dtype=np.float32)

        D, I = self.index.search(query_embedding, k)

        retrieved_chunks = []
        for i in I[0]:
            if i != -1: # Ensure the index is valid
                retrieved_chunks.append(self.chunks_metadata[i])
        return retrieved_chunks

    # Your get_answer method should call retrieve_chunks and then use Gemini.
    # It should NOT directly use an embedding model itself.
    def get_answer(self, query):
        retrieved_chunks = self.retrieve_chunks(query)

        if not retrieved_chunks:
            return "No relevant information found for your query in the documents."

        # Construct the context for Gemini
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])

        # Use Gemini to generate the answer based on the query and context
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""
            You are a helpful travel assistant. Based on the following context, answer the user's query.
            If the context does not contain enough information to answer the question, state that you cannot answer from the provided context.

            Context:
            {context}

            User Query: {query}

            Answer:
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return f"An error occurred while generating the answer: {str(e)}"