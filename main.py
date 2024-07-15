from flask import Flask, jsonify, request
import os
# RECOMMENDATION
import torch
from transformers import BertTokenizer, BertModel
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def embed_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def recommend(resume, jobD, threshold=0.5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    job_descriptions = [preprocess_text(desc) for desc in jobD.values()]
    resume_text = preprocess_text(resume)

    resume_embedding = embed_text(resume_text, tokenizer, model)
    job_embeddings = [embed_text(desc, tokenizer, model) for desc in job_descriptions]

    similarities = [cosine_similarity(resume_embedding, job_emb)[0][0] for job_emb in job_embeddings]
    recommendations = [(list(jobD.keys())[i], list(jobD.values())[i]) for i in range(len(similarities)) if similarities[i] >= threshold]

    return dict(recommendations)



@app.route('/recommend/',methods=['GET'])
def recommendation():
    resume=request.args.get('resume')
    jds=request.args.get('jobdescription')
    if not resume or not jds:
        return jsonify({'Error':'No Data Entered'})
    threshold=0.85
    return jsonify(recommend(resume,jds,threshold))

if __name__ == '__main__':
    app.run()
