from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import traceback

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
        except Exception as e:
            print(f"--- FAILED TO LOAD CSV: {e} ---")
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
CORS(app)

dataset_filename = 'rankings.csv'
qa_system = UniversityQASystem(dataset_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        university_name = data.get('university_name')
        if not university_name or qa_system.df is None:
            return jsonify({'error': 'Dataset not loaded properly.'}), 400
        
        details = qa_system.get_university_details(university_name)
        return jsonify(details)
    except Exception as e:
        print("--- AN ERROR OCCURRED ON THE SERVER ---")
        traceback.print_exc()
        print("------------------------------------")
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)