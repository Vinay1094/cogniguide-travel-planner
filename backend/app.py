import os
from flask_cors import CORS
from flask import Flask, request, jsonify

# Import the get_rag_response function from your rag_pipeline
# We need to adjust the import path because app.py is in 'backend'
# relative to rag_app.py in 'rag_pipeline'.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_pipeline')))
from rag_app import get_rag_response

app = Flask(__name__) 
CORS(app)


@app.route('/')
def home():
    return "CogniGuide Backend API is running!"

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query']
    print(f"Received API query: {user_query}")

    try:
        # Call your RAG function
        response = get_rag_response(user_query)
        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error processing RAG query: {e}")
        return jsonify({"error": "An internal error occurred while processing your request."}), 500

# ... previous code ...

# Line 37: The if statement
if __name__ == '__main__':
# Line 38: This line might be blank or not indented,
# Line 39: or the next line of code (like app.run()) isn't indented.
    # This is how it should look: the app.run() line is indented 4 spaces
    app.run(debug=True) # or app.run(host='0.0.0.0', port=5000)
