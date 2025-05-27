import streamlit as st
import joblib
import pandas as pd
import csv

# Load models
clf = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
df = pd.read_csv("customer_queries.csv")

def chatbot_response(user_query):
    vect_input = vectorizer.transform([user_query])
    category = clf.predict(vect_input)[0]
    response = df[df['category'] == category]['response'].sample(1).values[0]
    return response, category

st.title("🤖 AI Customer Support Chatbot")
user_input = st.text_input("Ask your question:")

if user_input:
    response, category = chatbot_response(user_input)
    st.markdown(f"**Bot:** {response}")
    st.markdown(f"*Category: {category}*")

    feedback = st.radio("Was this helpful?", ("Yes", "No"))

    if st.button("Submit Feedback"):
        with open("feedback_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_input, category, response, feedback])
        st.success("✅ Thank you for your feedback!")
