import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Fix for missing punkt and stopwords
nltk.data.path.append('nltk_data')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='nltk_data')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='nltk_data')

ps = PorterStemmer()

def transform_text(text):
    text= text.lower()
    text= nltk.word_tokenize(text)
    y=[]
    #remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    # remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
     # stemming
    for i in text:
        y.append(ps.stem(i)) 

    return " ".join(y)




tfidf= pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('mnb_model.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_text = st.text_area("Enter the message")

if st.button("Predict"):

    # 1. preprocess the text
    transformed_text = transform_text(input_text)

    # 2. vectorize the text
    vectorized_text = tfidf.transform([transformed_text])

    # 3. make predictions
    prediction = model.predict(vectorized_text)[0]

    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")   