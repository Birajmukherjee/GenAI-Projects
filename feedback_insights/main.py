import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from transformers import pipeline

def run():
    df = pd.read_csv('feedback.csv')

    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)  # CPU mode

    df['sentiment'] = df['comments'].apply(lambda x: sentiment_analyzer(x)[0]['label'])

    display_wordcloud(df)

    with st.expander("Sentiment Distribution"):
        st.write("### Sentiment Distribution of Feedback")
        sentiment_pie_chart(df)

    with st.expander("Average Rating by Role"):
        st.write("### Average Rating by Role")
        average_rating_line_chart(df)


def sentiment_pie_chart(df):
    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_ylabel('')
    plt.title("Sentiment Distribution")
    st.pyplot(fig)

def average_rating_line_chart(df):
    fig, ax = plt.subplots()
    avg_rating_by_role = df.groupby('role')['rating'].mean().reset_index()

    sns.lineplot(x='role', y='rating', data=avg_rating_by_role, marker='o', sort=False, ax=ax, linewidth=2.5)

    ax.set_xlabel("Role")
    ax.set_ylabel("Average Rating")
    plt.title("Average Rating by Role", fontsize=14)
    
    st.pyplot(fig)

def display_wordcloud(df):
    comments_text = " ".join(df['comments'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.title("Word Cloud for Feedback Comments")
    st.pyplot(fig)

if __name__ == "__main__":
    st.title("Feedback Analytics")
    run()
