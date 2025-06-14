# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Ensure the rag_pipeline directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_pipeline')))
from rag import RAGPipeline

app = Flask(__name__)
CORS(app)

# TEMPORARILY COMMENT OUT THIS LINE - WE'LL MOVE IT FOR DEBUGGING
# rag_pipeline = RAGPipeline() # Or however you instantiate it

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
        # TEMPORARILY MOVE THE RAGPIPELINE INITIALIZATION HERE
        # This will let the app start, but might crash on the first /ask request
        rag_pipeline = RAGPipeline() 

        answer = rag_pipeline.get_answer(query)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error processing query: {e}") # This should now print the specific error!
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))