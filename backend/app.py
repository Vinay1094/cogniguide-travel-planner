# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS # Add this line
import os
import sys

# Ensure the rag_pipeline directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_pipeline')))
from rag_app import RAGPipeline # Adjust import if your RAGPipeline class is named differently

app = Flask(__name__)
CORS(app) # Add this line right after app = Flask(__name__)

# Initialize RAG Pipeline globally to avoid re-loading on each request
# This assumes faiss_index.bin and chunks_metadata.json are in the rag_pipeline directory
rag_pipeline = RAGPipeline() # Or however you instantiate it

@app.route('/')
def home():
    return "CogniGuide Backend is Running!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        answer = rag_pipeline.get_answer(query)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error processing query: {e}") # This will show in Render logs!
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Use Gunicorn's default host/port when running directly for local testing
    # For Render deployment, Gunicorn will handle host/port
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))