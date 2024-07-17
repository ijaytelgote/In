from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def recommend(resume_text, job_descriptions, threshold=0.85):
    # Extract job descriptions and titles from dictionary
    titles = list(job_descriptions.keys())
    jds = list(job_descriptions.values())
    
    # Vectorize job descriptions and resume
    vectorizer = TfidfVectorizer(stop_words='english')
    job_matrix = vectorizer.fit_transform(jds + [resume_text])
    
    # Separate the vectorized resume
    resume_vector = job_matrix[-1]
    job_matrix = job_matrix[:-1]
    
    # Calculate cosine similarity between resume and each job description
    similarities = cosine_similarity(job_matrix, resume_vector).flatten()
    
    # Find job titles with similarity above threshold
    recommended_jobs = [titles[i] for i, sim in enumerate(similarities) if sim >= threshold]
    
    return recommended_jobs

@app.route('/recommends/', methods=['POST'])
def recommendation():
    data = request.get_json()
    if not data:
        return jsonify({'Error': 'No data received'}), 400
    
    resume = data.get('resume')
    job_descriptions = data.get('job_descriptions', {})
    
    if not resume:
        return jsonify({'Error': 'Resume not provided'}), 400
    if not job_descriptions:
        return jsonify({'Error': 'Job descriptions not provided'}), 400
    
    try:
        threshold = float(data.get('threshold', 0.85))
    except ValueError:
        return jsonify({'Error': 'Invalid threshold value'}), 400
    
    recommended_jobs = recommend(resume, job_descriptions, threshold)
    
    return jsonify({'recommendations': recommended_jobs})

if __name__ == '__main__':
    app.run(debug=True)
