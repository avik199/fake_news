import re
import string
import pickle
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

model = pickle.load(open('model.pkl','rb'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    porter = PorterStemmer()
    stemmed_tokens = [porter.stem(word) for word in filtered_tokens]

    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text


def predict_news_authenticity(news_text):
    preprocessed_news_text = preprocess_text(news_text)
    prediction = model.predict([preprocessed_news_text])

    if prediction == 0:
        return "Fake"
    else:
        return "Real"



st.title("Fake News Detection")

input_news = st.text_input("Enter the News")

if st.button('Predict'):

    # 1. preprocess
    result=predict_news_authenticity(input_news)
    # 4. Display
    if result == "Fake":
        st.header("The news is fake")
    else :
        st.header("The news is true")
    print("The news is:", result)
if st.button('Clear Output'):
    st.header("")