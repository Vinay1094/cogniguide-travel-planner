import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss # For efficient similarity search

# Define paths
PROCESSED_CHUNKS_PATH = './processed_chunks/travel_chunks.json'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A good small, general-purpose model
FAISS_INDEX_PATH = './faiss_index.bin' # Path to save the FAISS index
CHUNKS_METADATA_PATH = './chunks_metadata.json' # Path to save chunk text and metadata

def create_embeddings_and_faiss_index():
    print(f"Loading chunks from {PROCESSED_CHUNKS_PATH}...")

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve the PROCESSED_CHUNKS_PATH relative to the script's directory
    resolved_chunks_path = os.path.abspath(os.path.join(script_dir, PROCESSED_CHUNKS_PATH))

    if not os.path.exists(resolved_chunks_path):
        print(f"Error: Processed chunks file not found at {resolved_chunks_path}")
        print("Please ensure you've successfully run preprocess_data.py first.")
        return

    with open(resolved_chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in the file. Please check your data/preprocessing.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
    # This line will download the model the first time it runs
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")

    # Extract text from chunks for embedding
    chunk_texts = [chunk['text'] for chunk in chunks]

    print("Generating embeddings for chunks...")
    # This generates embeddings in batches for efficiency
    embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)
    print("Embeddings generated.")
    print(f"Shape of embeddings: {embeddings.shape}") # Should be (num_chunks, embedding_dimension)

    # Store chunks' text and metadata separately
    # This is important because FAISS only stores vectors, not the original text
    chunks_metadata = [
        {"text": chunk['text'], "source": chunk.get('source', 'unknown'), "chunk_id": chunk.get('chunk_id', idx)}
        for idx, chunk in enumerate(chunks)
    ]

    # Resolve path for metadata file
    resolved_metadata_path = os.path.abspath(os.path.join(script_dir, CHUNKS_METADATA_PATH))
    # Save chunks metadata
    with open(resolved_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_metadata, f, indent=4)
    print(f"Saved chunk metadata to {resolved_metadata_path}")

    # Create a FAISS index
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension) # L2 distance is common for similarity
    index.add(embeddings) # Add the embeddings to the index

    print(f"FAISS index created with {index.ntotal} vectors.")

    # Resolve path for FAISS index file
    resolved_faiss_index_path = os.path.abspath(os.path.join(script_dir, FAISS_INDEX_PATH))
    # Save the FAISS index
    faiss.write_index(index, resolved_faiss_index_path)
    print(f"FAISS index saved to {resolved_faiss_index_path}")

if __name__ == "__main__":
    create_embeddings_and_faiss_index()
    print("Embedding process complete.")