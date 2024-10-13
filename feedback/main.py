import streamlit as st
from datetime import datetime
import pandas as pd
import os

def save_feedback_to_csv(feedback_data):
    csv_file = os.path.join(os.getcwd(), 'feedback.csv')
    file_exists = os.path.isfile(csv_file)
    df = pd.DataFrame([feedback_data])
    df.to_csv(csv_file, mode='a', index=False, header=not file_exists)


def open_feedback_dialog():
    st.markdown("<h5 style='text-align: center; color: #4CAF50;'>We Value Your Feedback</h5>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #828282;'>Your input helps us improve the quality of our services.</p>", unsafe_allow_html=True)

    with st.form(key="feedback_form"):
        name=st.text_input("Enter your full name")
        role=st.text_input("Enter your role (e.g., Engineer, Manager)")
        rating = st.radio("How would you rate your experience?", options=[1, 2, 3, 4, 5], index=4, format_func=lambda x: '⭐' * x)
        comments=st.text_area("Share your thoughts with us...")
        form_submit_button = st.form_submit_button("Submit")
    
    if form_submit_button:
        print(name)
        feedback_data = {
                        "name": name,
                        "role": role,
                        "rating": rating,
                        "comments": comments,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
        save_feedback_to_csv(feedback_data)
        st.markdown("<h5 style='text-align: center; color: #4CAF50;'>Your feedback is recoreded, Thank you!</h5>", unsafe_allow_html=True)

def run():
    open_feedback_dialog()