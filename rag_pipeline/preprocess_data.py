import os
import json
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define paths
DATA_DIR = '../data' # Path to your raw data
PROCESSED_DIR = './processed_chunks' # Where to save processed chunks

# --- ADD THESE DEBUGGING LINES HERE ---
# Get the directory where the current script is located (this will be 'rag_pipeline')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Resolve the DATA_DIR path relative to the script's directory
# This is the full, absolute path that Python is trying to access for your data
resolved_data_dir = os.path.abspath(os.path.join(script_dir, DATA_DIR))

print(f"DEBUG: Script is located at: {script_dir}")
print(f"DEBUG: Attempting to access data directory at: {resolved_data_dir}")
print(f"DEBUG: Does the data directory exist according to Python? {os.path.exists(resolved_data_dir)}")
# --- END OF DEBUGGING LINES ---

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    """Basic text cleaning (you might need more advanced cleaning based on your data)."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split()) # Remove extra spaces
    # Add more cleaning steps if necessary (e.g., HTML tag removal)
    return text

def chunk_text(text, source_filename, chunk_size=300, overlap_size=50):
    """
    Splits text into chunks using spaCy for sentence segmentation,
    then combines sentences into approximate chunk_size with overlap.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_id_counter = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split()) # Approximate token count
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Save current chunk
            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk).strip(),
                    "source": source_filename,
                    "chunk_id": f"{source_filename}_{chunk_id_counter}"
                })
                chunk_id_counter += 1
            
            # Start new chunk with overlap
            current_chunk = []
            if overlap_size > 0:
                # Add sentences from the end of the previous chunk as overlap
                overlap_sentences_start_idx = max(0, len(sentences) - (overlap_size // (chunk_size / len(sentences))) ) # Rough calculation
                
                # A more robust overlap would be:
                # Find how many tokens from the end of the last chunk to carry over
                # This simple loop adds last few sentences until overlap_size is met
                overlap_tokens_count = 0
                for j in reversed(range(len(current_chunk))):
                    if overlap_tokens_count < overlap_size:
                        current_chunk.insert(0, current_chunk[j]) # Add to start of new chunk
                        overlap_tokens_count += len(current_chunk[j].split())
                    else:
                        break
                
            current_chunk.append(sentence)
            current_length = len(" ".join(current_chunk).split())
            
    # Add the last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk).strip(),
            "source": source_filename,
            "chunk_id": f"{source_filename}_{chunk_id_counter}"
        })

    return chunks

def process_all_data(data_dir, processed_dir):
    all_processed_chunks = []

    # --- NEW DEBUGGING PRINT HERE ---
    files_found = os.listdir(resolved_data_dir) # Use the absolute path here
    print(f"DEBUG: Files found in resolved data directory: {files_found}")
    # --- END NEW DEBUGGING PRINT ---

    for filename in files_found: # Iterate through the files found
        filepath = os.path.join(resolved_data_dir, filename) # Join with absolute path

        # Ensure it's a file and ends with .txt or .md
        if os.path.isfile(filepath) and filename.endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)
            chunks = chunk_text(cleaned_text, filename)
            all_processed_chunks.extend(chunks)
            print(f"Processed {len(chunks)} chunks from {filename}")
        else:
            print(f"DEBUG: Skipping {filename} (not a .txt or .md file or is a directory)")

    # Save all chunks to a single JSON file
    output_filepath = os.path.join(processed_dir, 'travel_chunks.json')
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_processed_chunks, f, indent=4)
    print(f"Saved {len(all_processed_chunks)} total chunks to {output_filepath}")

if __name__ == "__main__":
    print("Starting data preprocessing and chunking...")
    process_all_data(DATA_DIR, PROCESSED_DIR)
    print("Preprocessing complete.")