import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import spacy
import pickle
import pandas as pd

import string
from nltk.corpus import stopwords


from nltk.stem import WordNetLemmatizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Streamlit interface
st.title("Financial News Search Engine")
query = st.text_input("Enter your search query:")

def text_transform(text):
    text = str(text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    lemmatizer = WordNetLemmatizer()
    
    for i in text:
        if i.isalnum() and (i not in stopwords.words('english') and i not in string.punctuation):
            i = lemmatizer.lemmatize(i)
            y.append(i)

    
    return " ".join(y) # return the list in the form of string    

vectorizer = pickle.load(open('vectorizer.pkl' , 'rb'))
tfidf_matrix = pickle.load(open('tfidf-matrix.pkl' , 'rb'))
df = pd.read_csv('main_cnbc.csv')

def search_query(query , tfidf_matrix , vectorizer):
    preprocessed_query = text_transform(query)
    query_vector = vectorizer.transform([preprocessed_query])
    
    similarity_scores = cosine_similarity(query_vector , tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1]
    
    return df.iloc[top_indices[:5]]
    

if query:
    search_results = search_query(query, tfidf_matrix, vectorizer)
    for index, row in search_results.iterrows():
        st.write(f"**Title**: {row['Headlines']}")
        st.write(f"**Description**: {row['Description']}")
        st.write("---")
