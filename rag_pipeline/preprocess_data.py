import os
import json
import numpy as np
import faiss
import re
import google.generativeai as genai # For Google embeddings

# Configure Google API key (make sure it's set as an environment variable)
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_CHUNKS_DIR = os.path.join(os.path.dirname(__file__), 'processed_chunks')
TRAVEL_CHUNKS_PATH = os.path.join(PROCESSED_CHUNKS_DIR, 'travel_chunks.json')
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index.bin')
CHUNKS_METADATA_PATH = os.path.join(os.path.dirname(__file__), 'chunks_metadata.json')

# --- Debugging Utility ---
def debug_log(message):
    print(f"DEBUG: {message}")

# --- Text Preprocessing and Chunking ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single
    text = text.strip()
    return text

def chunk_text(text, max_chunk_size=500):
    chunks = []
    current_chunk = []
    current_length = 0

    sentences = re.split(r'(?<=[.!?])\s+', text) # Split by sentences

    for sentence in sentences:
        # If adding the next sentence exceeds max_chunk_size, start a new chunk
        if current_length + len(sentence) + 1 > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1 # +1 for the space

    if current_chunk: # Add any remaining chunk
        chunks.append(" ".join(current_chunk))
    return chunks

def load_and_chunk_data(data_dir):
    all_chunks = []
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return []
    
    debug_log(f"Attempting to access data directory at: {data_dir}")
    debug_log(f"Does the data directory exist according to Python? {os.path.exists(data_dir)}")

    files_in_data_dir = os.listdir(data_dir)
    debug_log(f"Files found in resolved data directory: {files_in_data_dir}")

    for filename in files_in_data_dir:
        file_path = os.path.join(data_dir, filename)
        
        # Skip directories, .gitkeep, and non-text files
        if os.path.isdir(file_path) or not (filename.endswith('.txt') or filename.endswith('.md')):
            debug_log(f"Skipping {filename} (not a .txt or .md file or is a directory)")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            cleaned_content = clean_text(content)
            chunks = chunk_text(cleaned_content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{filename}_{i}",
                    'text': chunk,
                    'source': filename
                })
            print(f"Processed {len(chunks)} chunks from {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    return all_chunks

# --- Embedding Generation ---
def get_embedding(text_chunk):
    try:
        # Use Google's embedding model
        response = genai.embed_content(model="models/embedding-001", content=text_chunk)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding for chunk: {e}")
        return None

# --- Main Preprocessing Logic ---
if __name__ == '__main__':
    debug_log(f"Script is located at: {os.path.dirname(__file__)}")

    # Ensure processed_chunks directory exists
    os.makedirs(PROCESSED_CHUNKS_DIR, exist_ok=True)

    print("Starting data preprocessing and chunking...")
    travel_chunks_data = load_and_chunk_data(DATA_DIR)
    
    if travel_chunks_data:
        # Save raw chunks for potential review/reuse
        with open(TRAVEL_CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(travel_chunks_data, f, indent=4)
        print(f"Saved {len(travel_chunks_data)} total chunks to {TRAVEL_CHUNKS_PATH}")

        print("Generating embeddings and building FAISS index...")
        chunk_texts = [chunk['text'] for chunk in travel_chunks_data]
        
        # Generate embeddings in batches if many chunks to avoid API limits/timeouts
        embeddings = []
        metadata = []
        for i, chunk_text in enumerate(chunk_texts):
            print(f"Generating embedding for chunk {i+1}/{len(chunk_texts)}...")
            embedding = get_embedding(chunk_text)
            if embedding is not None:
                embeddings.append(embedding)
                metadata.append(travel_chunks_data[i])
            else:
                print(f"Skipping chunk {travel_chunks_data[i]['id']} due to embedding error.")

        if not embeddings:
            print("No embeddings generated. FAISS index cannot be built.")
        else:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            embedding_dim = embeddings_np.shape[1]

            # Build FAISS index
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embeddings_np)
            
            # Save FAISS index
            faiss.write_index(index, FAISS_INDEX_PATH)
            print(f"FAISS index built and saved to {FAISS_INDEX_PATH} with dimension {embedding_dim}.")

            # Save chunks metadata (with IDs linked to index)
            with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            print(f"Chunks metadata saved to {CHUNKS_METADATA_PATH}.")

    else:
        print("No chunks processed. FAISS index not built.")

    print("Preprocessing complete.")