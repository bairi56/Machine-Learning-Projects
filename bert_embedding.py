import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

import numpy as np

def measure_relevance(user_answer, expected_answer):
    user_vec = get_bert_embeddings(user_answer)
    expected_vec = get_bert_embeddings(expected_answer)
    similarity = np.dot(user_vec, expected_vec.T) / (np.linalg.norm(user_vec) * np.linalg.norm(expected_vec))
    return similarity[0][0]

data = {
    'text': [
        "Tell me about yourself.", 
        "Why do you want this job?", 
        "What are your strengths?", 
        "Describe a challenging project you worked on."
    ], 
    'expected_answers': [
        "I am a software developer with 5 years of experience in software development.", 
        "I am interested in this job because it allows me to use my skills to contribute to exciting projects.", 
        "My strengths include problem-solving, teamwork, and adaptability.", 
        "I worked on a project where I had to develop a complex software system under a tight deadline."
    ]
}

questions = data['text']
expected_answers = data['expected_answers']

for i, question in enumerate(questions):
    user_answer = input(question + "\n")
    relevance = measure_relevance(user_answer, expected_answers[i])
    print(f"Relevance Score: {relevance:.2f}\n")
