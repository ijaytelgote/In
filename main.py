!python -m spacy download en_core_web_sm
import string

import nltk
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pendulum
import spacy
from flask import Flask, jsonify, request
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')

app = Flask(__name__)



def extract_skills(text):
    skills_pattern = (
        r"\b(Skill(?:s|z)?|Abilit(?:ies|y|tys)?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Technological Skills?|TECHNICAL SKILLS?|Technical Expertise)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Experience|Experiences|Employment|Work History|Professional Background|Projects|its last|Project Work|Case Studies|Education|Educations|Academic Background|Qualifications|Studies|Soft Skills|Achievements|$))"
    )
    skills_match = re.search(skills_pattern, text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        return skills_match.group(2).strip()
    return None

def extract_experience(text):
    experience_pattern = (
        r"\b(Experience|Experiences|Employments?|Work History|Professional Background|Career History|Professional Experience|Job History|Work Experience|Job Experiences?|Employment History|Work Experiences?|Professional Experiences?|WORK EXPERIENCE)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Projects?|Project Work|Case Studies|Education|Educations|Academic Background|Qualifications|Studies|its last|Soft Skills|Achievements|$))"
    )
    experience_match = re.search(experience_pattern, text, re.DOTALL | re.IGNORECASE)
    if experience_match:
        return experience_match.group(2).strip()
    return None

def extract_education(text):
    education_pattern = (
        r"\b(Education|Educations|Academic Background|Qualifications|Studies|Academic Qualifications|Educational Background|Academic History|Educational History|Education and Training|Educational Qualifications|EDUCATION)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Projects?|Project Work|Case Studies|its last|Soft Skills|Achievements|$))"
    )
    education_match = re.search(education_pattern, text, re.DOTALL | re.IGNORECASE)
    if education_match:
        return education_match.group(2).strip()
    return None

def extract_projects(text):
    projects_pattern = (
        r"\b(Projects?|Project Work|Case Studies|Project Experience|Key Projects|Notable Projects|Significant Projects|Project Undertakings?|Project Initiatives?|Major Projects|Project Details?|PROJECTS?|Project Assignments?|Project Highlights)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Education|Educations|Academic Background|Qualifications|Studies|its last|Soft Skills|Achievements|$))"
    )
    projects_match = re.search(projects_pattern, text, re.DOTALL | re.IGNORECASE)
    if projects_match:
        return projects_match.group(2).strip()
    return None

def extract_soft_skills(text):
    soft_skills_pattern = (
        r"\b(Soft Skills|Personal Skills|Interpersonal Skills|Core Skills|SOFT SKILLS)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Projects?|Project Work|Case Studies|Education|Educations|Academic Background|Qualifications|Studies|Achievements|its last|$))"
    )
    soft_skills_match = re.search(soft_skills_pattern, text, re.DOTALL | re.IGNORECASE)
    if soft_skills_match:
        return soft_skills_match.group(2).strip()
    return None

def extract_achievements(text):
    achievements_pattern = (
        r"\b(Achievements?|Accomplishments?|Awards?|Honors?|ACHIEVEMENTS)\b"
        r"[\s:\-\n]*"
        r"(.+?)(?=\b(Skills?|Abilities?|Competenc(?:ies|y)|Expertise|Skillset|Technical Skills?|Technical Abilities?|Experience|Experiences|Employment|Work History|Professional Background|Projects?|Project Work|Case Studies|Education|Educations|Academic Background|Qualifications|Studies|Soft Skills|its last|$))"
    )
    achievements_match = re.search(achievements_pattern, text, re.DOTALL | re.IGNORECASE)
    if achievements_match:
        return achievements_match.group(2).strip()
    return None

def extract_name(resume_text):
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)
    
    # Define the pattern
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern])
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text.replace('{', '').replace('}', '')

def parse_date(date_str):
    try:
        parsed_date = pendulum.parse(date_str, strict=False)
        return parsed_date
    except ValueError:
        raise ValueError(f"No valid date format found for '{date_str}'")

def calculate_experience(start_date, end_date):
    duration = end_date.diff(start_date)
    years = duration.years
    months = duration.months
    return years + months / 12

def calculate_total_experience(resume_text):
    # Regular expression to match date ranges with various formats including year-only ranges
    date_range_pattern = re.compile(
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\'\"`*+,\-–.:/;!@#$%^&(){}\[\]<>_=~`]*\d{2,4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\d{4})\s*(?:[-–to ]+)\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\'\"`*+,\-–.:/;!@#$%^&(){}\[\]<>_=~`]*\d{2,4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{4}|\d{4}|\b[Tt]ill\b|\b[Nn]ow\b|\b[Pp]resent\b|\b[Oo]ngoing\b|\b[Cc]ontinue\b|\b[Cc]urrent\b)?'
    )

    date_matches = date_range_pattern.findall(resume_text)

    total_experience = 0
    
    for start_date_str, end_date_str in date_matches:
        try:
            start_date = parse_date(start_date_str.strip())
            end_date = pendulum.now() if not end_date_str or end_date_str.strip().lower() in ['till', 'now', 'present', 'ongoing', 'continue', 'current'] else parse_date(end_date_str.strip())
            
            experience = calculate_experience(start_date, end_date)
            total_experience += experience
        except ValueError as e:
            print(e)
    
    return str(round(total_experience, 2)) + ' Years'

def parsed(resume1):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{2,4}[-.\s]?\d{10}|\d{10}|\d{11})'
    
    email_match = re.search(email_pattern, resume1)
    phone_match = re.search(phone_pattern, resume1)

    email = email_match.group() if email_match else None
    phone = phone_match.group() if phone_match else None

    name = extract_name(resume1)
    skills = extract_skills(resume1)
    experience = extract_experience(resume1)
    education = extract_education(resume1)
    projects = extract_projects(resume1)
    soft_skills = extract_soft_skills(resume1)
    achievements = extract_achievements(resume1)
    
    experience_years = calculate_total_experience(experience) if experience else "0 Years"
    
    return {
        'Name': name,
        'Email': email,
        'Phone': phone,
        'Skills': skills,
        'Experience': experience,
        'Education': education,
        'Projects': projects,
        'Soft Skills': soft_skills,
        'Achievements': achievements,
        'Total Experience': experience_years
    }


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

@app.route('/parse', methods=['POST'])
def parse_resume():
    data = request.json
    resume_text = data.get('resume_text')
    if not resume_text:
        return jsonify({"error": "No resume text provided"}), 400
    
    parsed_data = parsed(resume_text)
    return jsonify(parsed_data)



if __name__ == '__main__':
    app.run(debug=True)
