import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl','rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl','rb') as file:
    vectorizer = pickle.load(file)


st.title('Sentiments  Analysis Application')


user_input = st.text_area("Enter text to analyze sentiment:")

if st.button('Predict'):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        st.write(f"**Predicted Sentiment:** {prediction.capitalize()}")