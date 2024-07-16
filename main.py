from flask import Flask, jsonify, request
import re
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize tokenizer and model outside the recommend function for efficiency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def recommend(resume, jds, threshold=0.5):
    job_descriptions = [preprocess_text(desc) for desc in jds.split(';')]
    resume_text = preprocess_text(resume)

    resume_embedding = embed_text(resume_text, tokenizer, model)
    job_embeddings = [embed_text(desc, tokenizer, model) for desc in job_descriptions]

    similarities = [cosine_similarity(resume_embedding, job_emb)[0][0] for job_emb in job_embeddings]
    recommendations = [(desc, sim) for desc, sim in zip(job_descriptions, similarities) if sim >= threshold]

    return recommendations

@app.route('/recommend/', methods=['GET'])
def recommendation():
    resume = request.args.get('resume')
    jds = request.args.get('jobdescription')
    if not resume or not jds:
        return jsonify({'Error': 'No Data Entered'})

    threshold = 0.85
    recommendations = recommend(resume, jds, threshold)

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run()
