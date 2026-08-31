import streamlit as st
import pandas as pd

st.title("AI-Based Literature Review Assistant")
st.write(
    "An AI-based system for finding relevant "
    "research papers using Natural Language Processing."
)

st.info(
    "Enter a research topic below to find "
    "the most relevant papers from the dataset."
)
df = pd.read_csv("AI_Literature_Dataset.csv")
st.metric("Research Papers in Dataset", len(df))
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
# Search interface

st.header("Search Research Papers")

query = st.text_input(
    "Enter your research topic:",
    placeholder="Example: machine learning in healthcare"
)

top_n = st.slider(
    "Number of papers to show:",
    min_value=1,
    max_value=10,
    value=5
)

min_year = st.number_input(
    "Minimum publication year:",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=int(df["Year"].min()),
    step=1
)

if st.button("Search Papers"):

    if query.strip():

        results = search_papers(query, top_n)

        results = results[results["Year"] >= min_year]

        if results.empty:
            st.warning("No papers found for the selected year.")

        else:
            st.subheader("Relevant Papers")

            for i, (_, row) in enumerate(results.iterrows(), 1):

                with st.expander(f"{i}. {row['Title']}"):

                    st.write("**Year:**", row["Year"])

                    st.write(
                        "**Relevance Score:**",
                        round(row["Relevance_Score"], 3)
                    )

                    st.write("**Abstract:**")
                    st.write(row["Abstract"])

                    st.write("**Quick Summary:**")

                    abstract = str(row["Abstract"])
                    sentences = abstract.split(".")
                    summary = ". ".join(sentences[:2])

                    st.write(summary + ".")

                    st.write("**Paper:**")
                    st.write(row["URL"])

            csv = results.to_csv(index=False)

            st.download_button(
                "Download Results",
                csv,
                "search_results.csv",
                "text/csv"
            )
                

            paper_titles = results["Title"].tolist()

            if len(paper_titles) >= 2:

                selected_papers = st.multiselect(
    "Select 2 papers to compare:",
    paper_titles,
    max_selections=2,
    key="paper_comparison"
)
                if len(selected_papers) == 2:

                    paper1 = results[
                        results["Title"] == selected_papers[0]
                    ].iloc[0]

                    paper2 = results[
                        results["Title"] == selected_papers[1]
                    ].iloc[0]

                    st.write("### Paper 1")
                    st.write("**Title:**", paper1["Title"])
                    st.write("**Year:**", paper1["Year"])
                    st.write(
                        "**Relevance Score:**",
                        round(paper1["Relevance_Score"], 3)
                    )
                    st.write("**Abstract:**", paper1["Abstract"])

                    st.write("### Paper 2")
                    st.write("**Title:**", paper2["Title"])
                    st.write("**Year:**", paper2["Year"])
                    st.write(
                        "**Relevance Score:**",
                        round(paper2["Relevance_Score"], 3)
                    )
                    st.write("**Abstract:**", paper2["Abstract"])

            else:
                st.info(
                    "Search for at least 2 papers to compare them."
                )
    else:
        st.warning("Please enter a research topic.")

# Sidebar # Paper Comparison

st.header("Compare Research Papers")

if "results" in locals() and len(results) >= 2:

    paper_titles = results["Title"].tolist()

    selected_papers = st.multiselect(
        "Select 2 papers to compare:",
        paper_titles,
        max_selections=2
    )

    if len(selected_papers) == 2:

        paper1 = results[
            results["Title"] == selected_papers[0]
        ].iloc[0]

        paper2 = results[
            results["Title"] == selected_papers[1]
        ].iloc[0]

        st.subheader("Paper 1")
        st.write("**Title:**", paper1["Title"])
        st.write("**Year:**", paper1["Year"])
        st.write(
            "**Relevance Score:**",
            round(paper1["Relevance_Score"], 3)
        )
        st.write("**Abstract:**", paper1["Abstract"])

        st.subheader("Paper 2")
        st.write("**Title:**", paper2["Title"])
        st.write("**Year:**", paper2["Year"])
        st.write(
            "**Relevance Score:**",
            round(paper2["Relevance_Score"], 3)
        )
        st.write("**Abstract:**", paper2["Abstract"])

else:

    st.info("Search for papers first to enable comparison.")
        # Sidebar
st.sidebar.title("About")

st.sidebar.write(
    "This application helps users find "
    "relevant research papers using TF-IDF "
    "and cosine similarity."
)

st.sidebar.write("Dataset: 1,000 research papers")
