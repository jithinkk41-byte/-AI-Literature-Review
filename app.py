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
