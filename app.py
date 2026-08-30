import streamlit as st
import pandas as pd

st.title("AI-Based Literature Review Assistant")

# Load the dataset
df = pd.read_csv("AI_Literature_Dataset.csv")

# Show dataset information
st.write("Total number of papers:", len(df))

# Show first 5 papers
st.subheader("Sample Papers")
st.dataframe(df.head())
# Prepare text for searching
df["Title"] = df["Title"].fillna("")
df["Abstract"] = df["Abstract"].fillna("")

df["Search_Text"] = (
    df["Title"].astype(str) + " " +
    df["Abstract"].astype(str)
).str.lower()

st.success("Dataset is ready for searching!")
# Create TF-IDF search system
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(
    df["Search_Text"]
)

st.success("TF-IDF search system is ready!")
# Function to search relevant papers
from sklearn.metrics.pairwise import cosine_similarity

def search_papers(query, top_n=5):
    query_vector = vectorizer.transform([query.lower()])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices].copy()

    results["Relevance_Score"] = similarity_scores[top_indices]

    return results

st.success("Paper search function is ready!")
