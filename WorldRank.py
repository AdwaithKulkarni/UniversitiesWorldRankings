from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
import json
import traceback
import requests # Ensure requests is imported

pd.options.mode.chained_assignment = None

class UniversityQASystem:
    def __init__(self, csv_path):
        self.all_columns = [
            'RANK_2025', 'RANK_2024', 'Location', 'Region', 'SIZE',
            'FOCUS', 'RES.', 'STATUS', 'Academic_Reputation_Score',
            'Academic_Reputation_Rank', 'Employer_Reputation_Score',
            'Employer_Reputation_Rank', 'Faculty_Student_Score',
            'Faculty_Student_Rank', 'Citations_per_Faculty_Score',
            'Citations_per_Faculty_Rank', 'International_Faculty_Score',
            'International_Faculty_Rank', 'International_Students_Score',
            'International_Students_Rank', 'International_Research_Network_Score',
            'International_Research_Network_Rank', 'Employment_Outcomes_Score',
            'Employment_Outcomes_Rank', 'Sustainability_Score',
            'Sustainability_Rank', 'Overall_Score'
        ]
        self.df = self._load_data(csv_path)
        if self.df is not None:
            self._preprocess_data()

    def _load_data(self, csv_path):
        try:
            return pd.read_csv(csv_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Error: The file '{csv_path}' was not found.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading the file: {e}")
            return None

    def _preprocess_data(self):
        self.df.columns = [col.strip().replace(' ', '_') for col in self.df.columns]
        self.df['Institution_Name'] = self.df['Institution_Name'].astype(str).str.strip()
        for col in self.df.columns:
            if '_Score' in col or '_Rank' in col or 'RANK_' in col:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def find_university(self, query):
        query_lower = query.lower().strip()
        if not query_lower:
            return None

        exact_match = self.df[self.df['Institution_Name'].str.lower() == query_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]

        query_words = set(query_lower.split())

        def contains_all_words(uni_name):
            uni_words = set(str(uni_name).lower().split())
            return query_words.issubset(uni_words)

        word_based_match = self.df[self.df['Institution_Name'].apply(contains_all_words)]
        if not word_based_match.empty:
            return word_based_match.iloc[0]

        partial_match = self.df[self.df['Institution_Name'].str.lower().str.contains(query_lower, na=False)]
        if not partial_match.empty:
            return partial_match.iloc[0]

        return None

    def get_university_details(self, name):
        university = self.find_university(name)
        if university is None:
            return {'error': f"Sorry, could not find any university matching '{name}'."}

        details = {'Institution_Name': university['Institution_Name']}
        for col in self.all_columns:
            if col in university:
                value = university.get(col)
                if pd.isna(value):
                    details[col] = 'N/A'
                elif isinstance(value, np.integer):
                    details[col] = int(value)
                elif isinstance(value, np.floating):
                    details[col] = float(value)
                else:
                    details[col] = str(value)
        return details

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

dataset_filename = 'rankings.csv' # Make sure your CSV file is named rankings.csv
qa_system = UniversityQASystem(dataset_filename)

# Simple check to see if the message is likely a university query
def is_university_query(message):
    keywords = ['university', 'college', 'rank', 'score', 'location', 'reputation', 'employment', 'sustainability', 'citations', 'details of']
    message_lower = message.lower()
    if any(keyword in message_lower for keyword in keywords):
        return True
    # Also check if it looks like a direct university name (e.g., "Harvard University")
    # This is a heuristic and might need fine-tuning based on your data
    potential_uni_name = message_lower.replace('what is the', '').replace('tell me about', '').strip()
    if qa_system.find_university(potential_uni_name):
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat_query(): # Removed 'async' keyword
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'No message provided.'}), 400

        # Check if the message is a university-specific query
        if is_university_query(user_message):
            # Attempt to extract university name from the message
            uni_name_match = re.search(r'(?:of|for|about|tell me about|what is the|details of)\s+(.*?)(?:\?|\.|$)', user_message, re.IGNORECASE)
            if uni_name_match:
                university_name = uni_name_match.group(1).strip()
            else:
                # If no specific phrase, assume the whole message is the university name
                university_name = user_message.replace('?', '').strip()

            details = qa_system.get_university_details(university_name)
            if 'error' not in details:
                # If university found, return its details directly
                return jsonify(details)
            else:
                # If university not found by specific query, proceed to LLM for a conversational "not found" response
                pass # Fall through to LLM call below

        # If not a university query, or if university not found by data, use Gemini API for general chat
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {"contents": chat_history}
        apiKey = os.environ.get("GOOGLE_API_KEY", "") # Get API key from environment variable

        if not apiKey:
            print("--- GOOGLE_API_KEY environment variable not set! ---")
            return jsonify({"error": "AI service not configured. API Key missing."}), 500

        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

        print(f"--- Calling Gemini API for message: '{user_message}' ---")
        try:
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()
            print(f"--- Gemini API Raw Response: {json.dumps(result, indent=2)} ---")

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
                return jsonify({"response": ai_response})
            else:
                print("--- Unexpected Gemini API response structure. ---")
                return jsonify({"error": "Failed to get a valid response from the AI model."}), 500
        except requests.exceptions.RequestException as req_e:
            print(f"--- Error calling Gemini API: {req_e} ---")
            return jsonify({"error": f"Failed to connect to AI service: {str(req_e)}"}), 500
        except json.JSONDecodeError as json_e:
            print(f"--- JSON decoding error from Gemini API response: {json_e} ---")
            print(f"--- Raw response text: {response.text} ---")
            return jsonify({"error": "Invalid response from AI service."}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable, default to 5000 for local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)