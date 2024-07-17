import string

import nltk
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/match', methods=['POST'])
def match_resume_to_jobs():
    data = request.json
    resume = data['resume']
    job_descriptions_dict = data['job_descriptions']

    # Preprocess the resume
    resume = preprocess_text(resume)

    # Separate job titles and their descriptions
    job_titles = list(job_descriptions_dict.keys())
    job_descriptions = [preprocess_text(desc) for desc in job_descriptions_dict.values()]

    # Combine all texts for vectorization
    all_texts = [resume] + job_descriptions

    # Vectorize the texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Find the best match
    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_index]
    best_match_title = job_titles[best_match_index]

    response = {
        'best_match_title': best_match_title,
        'best_match_score': best_match_score,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
