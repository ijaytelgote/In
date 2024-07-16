from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def recommend(resume_text, job_descriptions, threshold=0.85):
    # Vectorize job descriptions and resume
    vectorizer = TfidfVectorizer(stop_words='english')
    job_matrix = vectorizer.fit_transform(job_descriptions + [resume_text])
    
    # Separate the vectorized resume
    resume_vector = job_matrix[-1]
    job_matrix = job_matrix[:-1]
    
    # Calculate cosine similarity between resume and each job description
    similarities = cosine_similarity(job_matrix, resume_vector).flatten()
    
    # Find job descriptions with similarity above threshold
    recommended_jobs = [job_descriptions[i] for i, sim in enumerate(similarities) if sim >= threshold]
    
    return recommended_jobs

@app.route('/recommend/', methods=['GET'])
def recommendation():
    resume = request.args.get('resume')
    jds = request.args.getlist('jobdescription')
    
    if not resume or not jds:
        return jsonify({'Error': 'No Data Entered'})
    
    threshold = float(request.args.get('threshold', 0.85))
    recommended_jobs = recommend(resume, jds, threshold)
    
    return jsonify({'recommendations': recommended_jobs})

if __name__ == '__main__':
    app.run(debug=True)
