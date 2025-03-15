import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer

#nltk.download('punkt')

model = pickle.load(open('model.pkl', 'rb'))
w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))

st.title('News prediction')

tt = st.text_input('Enter the news:')

st.write(tt)

stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def vectorize_text(tokens, model2, size):
    vectors = [model2.wv[word] for word in tokens if word in model2.wv]
    return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(size)

if st.button('Predict'):
    text_clean = preprocess_text(tt)

    tokens = nltk.word_tokenize(text_clean)


    vector_text = vectorize_text(tokens, w2v_model, 100)

    nv = []
    nv.append(vector_text)
    nv = np.array(nv)
    nv = nv.reshape(1, -1)

    pred = model.predict(nv)

    if pred[0][0]>0.5 and pred[0][1]<=0.5: st.write('Fake')
    if pred[0][1] > 0.5 and pred[0][0] <= 0.5: st.write('Real')
    print(pred)

    st.write(pred)
