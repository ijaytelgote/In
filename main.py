from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize an empty dictionary to store user-input job descriptions
user_job_descriptions = {}

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

@app.route('/add_job/', methods=['POST'])
def add_job():
    data = request.json
    title = data.get('title')
    description = data.get('description')
    
    if not title or not description:
        return jsonify({'Error': 'Title or Description missing in request body'})
    
    user_job_descriptions[title] = description
    return jsonify({'message': f'Job "{title}" added successfully'})

@app.route('/recommend/', methods=['GET'])
def recommendation():
    resume = request.args.get('resume')
    
    if not resume:
        return jsonify({'Error': 'No Resume Data Entered'})
    
    threshold = float(request.args.get('threshold', 0.85))
    recommended_jobs = recommend(resume, user_job_descriptions, threshold)
    
    return jsonify({'recommendations': recommended_jobs})

if __name__ == '__main__':
    app.run(debug=True)
