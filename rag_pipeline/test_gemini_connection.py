import os
import google.generativeai as genai

# Configure the API key from the environment variable
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("API key configured successfully.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it using: export GOOGLE_API_KEY='YOUR_KEY' (Mac/Linux) or set GOOGLE_API_KEY='YOUR_KEY' (Windows)")
    exit()

# --- THIS PART SHOULD BE COMMENTED OUT NOW ---
# print("\nListing available Gemini models:")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(f"- {m.name} (supports generateContent)")
#     else:
#         print(f"- {m.name} (does NOT support generateContent)")
# print("\n--- End of Model List ---")
# --- END COMMENTED OUT PART ---

# --- THESE LINES SHOULD BE UNCOMMENTED AND MODIFIED ---
model = genai.GenerativeModel('gemini-1.5-flash-latest') # <-- Use this model!

prompt = "Hello, Gemini! What is your purpose?"
print(f"\nSending prompt to Gemini: '{prompt}'")

try:
    response = model.generate_content(prompt)
    print("\nGemini's response:")
    print(response.text)
except Exception as e:
    print(f"\nAn error occurred while calling the Gemini API: {e}")
    print("Please check your API key, network connection, and Gemini API usage limits.")