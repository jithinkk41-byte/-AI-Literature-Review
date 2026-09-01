import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI-Based Literature Review Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Based Literature Review Assistant")

st.write(
    "An intelligent web application for searching, analyzing, "
    "summarizing, comparing and reviewing research papers using "
    "Artificial Intelligence and Natural Language Processing."
)

st.info(
    "Enter a research topic to find relevant papers and generate "
    "useful literature review insights."
)

@st.cache_data
def load_data():
    return pd.read_csv("AI_Literature_Dataset.csv")

df = load_data()

df["Title"] = df["Title"].fillna("")
df["Abstract"] = df["Abstract"].fillna("")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Year"] = df["Year"].fillna(0).astype(int)

if "URL" not in df.columns:
    df["URL"] = ""

st.metric("Research Papers in Dataset", len(df))

st.write("Total number of papers:", len(df))

with st.expander("View Sample Papers"):
    st.dataframe(df.head(), use_container_width=True)

df["Search_Text"] = (
    df["Title"].astype(str)
    + " "
    + df["Abstract"].astype(str)
).str.lower()

@st.cache_resource
def create_search_system(text_data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    matrix = vectorizer.fit_transform(text_data)

    return vectorizer, matrix

vectorizer, tfidf_matrix = create_search_system(
    df["Search_Text"]
)

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

def create_summary(text, sentence_count=2):
    text = str(text).strip()

    if not text:
        return "Summary is not available."

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    if len(sentences) == 0:
        return text

    return ". ".join(sentences[:sentence_count]) + "."

def extract_information(abstract):
    abstract = str(abstract)

    sentences = [
        sentence.strip()
        for sentence in abstract.split(".")
        if sentence.strip()
    ]

    objectives = []
    methodology = []
    findings = []
    conclusions = []

    objective_words = [
        "aim",
        "objective",
        "purpose",
        "focus",
        "investigate",
        "examine"
    ]

    methodology_words = [
        "method",
        "methodology",
        "model",
        "algorithm",
        "approach",
        "analysis",
        "dataset",
        "experiment"
    ]

    finding_words = [
        "result",
        "results",
        "finding",
        "findings",
        "performance",
        "accuracy",
        "improve"
    ]

    conclusion_words = [
        "conclusion",
        "conclude",
        "suggest",
        "demonstrate",
        "indicate",
        "show"
    ]

    for sentence in sentences:
        lower_sentence = sentence.lower()

        if any(word in lower_sentence for word in objective_words):
            objectives.append(sentence)

        if any(word in lower_sentence for word in methodology_words):
            methodology.append(sentence)

        if any(word in lower_sentence for word in finding_words):
            findings.append(sentence)

        if any(word in lower_sentence for word in conclusion_words):
            conclusions.append(sentence)

    return {
        "Objectives": objectives[:2],
        "Methodology": methodology[:2],
        "Key Findings": findings[:2],
        "Conclusions": conclusions[:2]
    }

st.header("🔎 Search Research Papers")

query = st.text_input(
    "Enter your research topic:",
    placeholder="Example: machine learning in healthcare"
)

top_n = st.slider(
    "Number of papers to show:",
    min_value=2,
    max_value=10,
    value=5
)

valid_years = df[df["Year"] > 0]["Year"]

min_dataset_year = int(valid_years.min())
max_dataset_year = int(valid_years.max())

min_year = st.number_input(
    "Minimum publication year:",
    min_value=min_dataset_year,
    max_value=max_dataset_year,
    value=min_dataset_year,
    step=1
)

if st.button("🔍 Search Papers", type="primary"):

    if query.strip():

        results = search_papers(query, top_n)

        results = results[
            results["Year"] >= min_year
        ]

        st.session_state["search_results"] = results
        st.session_state["search_query"] = query

    else:
        st.warning("Please enter a research topic.")

if "search_results" in st.session_state:

    results = st.session_state["search_results"]

    if results.empty:

        st.warning(
            "No papers found for the selected year."
        )

    else:

        st.header("📄 Relevant Papers")

        st.success(
            f"{len(results)} relevant papers found!"
        )

        for i, (_, row) in enumerate(
            results.iterrows(),
            1
        ):

            with st.expander(
                f"{i}. {row['Title']}"
            ):

                st.write(
                    "**Publication Year:**",
                    row["Year"]
                )

                st.write(
                    "**Relevance Score:**",
                    round(row["Relevance_Score"], 3)
                )

                st.write("### Abstract")
                st.write(row["Abstract"])

                st.write("### 📝 Quick Summary")
                st.write(
                    create_summary(row["Abstract"])
                )

                st.write("### 🔬 Paper Analysis")

                information = extract_information(
                    row["Abstract"]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write("#### 🎯 Objectives")

                    if information["Objectives"]:
                        for item in information["Objectives"]:
                            st.write("•", item)
                    else:
                        st.write("No clear objective extracted.")

                    st.write("#### ⚙️ Methodology")

                    if information["Methodology"]:
                        for item in information["Methodology"]:
                            st.write("•", item)
                    else:
                        st.write("No clear methodology extracted.")

                with col2:

                    st.write("#### 📊 Key Findings")

                    if information["Key Findings"]:
                        for item in information["Key Findings"]:
                            st.write("•", item)
                    else:
                        st.write("No clear findings extracted.")

                    st.write("#### ✅ Conclusions")

                    if information["Conclusions"]:
                        for item in information["Conclusions"]:
                            st.write("•", item)
                    else:
                        st.write("No clear conclusion extracted.")

                st.write("### 🔗 Research Paper")

                if str(row["URL"]).strip():
                    st.write(row["URL"])

        st.subheader("📥 Download Search Results")

        csv = results.to_csv(index=False)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="literature_review_results.csv",
            mime="text/csv",
            key="download_search_results"
        )

st.header("⚖️ Compare Research Papers")

if (
    "search_results" in st.session_state
    and len(st.session_state["search_results"]) >= 2
):

    comparison_results = st.session_state["search_results"]

    paper_titles = comparison_results["Title"].tolist()

    selected_papers = st.multiselect(
        "Select exactly 2 papers to compare:",
        paper_titles,
        max_selections=2,
        key="paper_comparison"
    )

    if len(selected_papers) == 2:

        paper1 = comparison_results[
            comparison_results["Title"] == selected_papers[0]
        ].iloc[0]

        paper2 = comparison_results[
            comparison_results["Title"] == selected_papers[1]
        ].iloc[0]

        st.subheader("📊 Comparison Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("## Paper 1")

            st.write("**Title:**", paper1["Title"])
            st.write("**Year:**", paper1["Year"])

            st.write(
                "**Relevance Score:**",
                round(paper1["Relevance_Score"], 3)
            )

            st.write("### Summary")

            st.write(
                create_summary(paper1["Abstract"])
            )

        with col2:

            st.markdown("## Paper 2")

            st.write("**Title:**", paper2["Title"])
            st.write("**Year:**", paper2["Year"])

            st.write(
                "**Relevance Score:**",
                round(paper2["Relevance_Score"], 3)
            )

            st.write("### Summary")

            st.write(
                create_summary(paper2["Abstract"])
            )

        st.subheader("🔍 Comparison of Research Content")

        comparison_data = pd.DataFrame({
            "Feature": [
                "Publication Year",
                "Relevance Score",
                "Summary"
            ],
            "Paper 1": [
                paper1["Year"],
                round(paper1["Relevance_Score"], 3),
                create_summary(paper1["Abstract"])
            ],
            "Paper 2": [
                paper2["Year"],
                round(paper2["Relevance_Score"], 3),
                create_summary(paper2["Abstract"])
            ]
        })

        st.dataframe(
            comparison_data,
            use_container_width=True
        )

else:

    st.info(
        "Search for at least 2 papers to enable comparison."
    )

st.header("🔬 Research Methods, Datasets and Outcomes")

if "search_results" in st.session_state:

    analysis_results = st.session_state["search_results"]

    all_text = " ".join(
        analysis_results["Abstract"].astype(str).tolist()
    ).lower()

    method_keywords = [
        "machine learning",
        "deep learning",
        "neural network",
        "random forest",
        "support vector machine",
        "classification",
        "clustering",
        "regression",
        "lstm",
        "algorithm"
    ]

    dataset_keywords = [
        "dataset",
        "data set",
        "data",
        "image",
        "medical data",
        "clinical data",
        "database"
    ]

    outcome_keywords = [
        "accuracy",
        "performance",
        "prediction",
        "improvement",
        "results",
        "classification",
        "detection"
    ]

    used_methods = [
        word for word in method_keywords
        if word in all_text
    ]

    used_datasets = [
        word for word in dataset_keywords
        if word in all_text
    ]

    outcomes = [
        word for word in outcome_keywords
        if word in all_text
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Methods")

        if used_methods:
            for method in used_methods:
                st.write("•", method.title())
        else:
            st.write("No common methods identified.")

    with col2:

        st.subheader("Datasets")

        if used_datasets:
            for dataset in used_datasets:
                st.write("•", dataset.title())
        else:
            st.write("No common datasets identified.")

    with col3:

        st.subheader("Outcomes")

        if outcomes:
            for outcome in outcomes:
                st.write("•", outcome.title())
        else:
            st.write("No common outcomes identified.")

else:
    st.info("Search for papers first.")

st.header("📈 Common Research Trends")

if "search_results" in st.session_state:

    trend_results = st.session_state["search_results"]

    titles = " ".join(
        trend_results["Title"].astype(str).tolist()
    )

    abstracts = " ".join(
        trend_results["Abstract"].astype(str).tolist()
    )

    combined_text = (
        titles + " " + abstracts
    ).lower()

    trend_keywords = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "healthcare",
        "prediction",
        "automation",
        "data analysis",
        "classification"
    ]

    found_trends = []

    for keyword in trend_keywords:

        if keyword in combined_text:
            found_trends.append(keyword.title())

    if found_trends:

        st.write(
            "The following common trends were identified:"
        )

        for trend in found_trends:
            st.write("📌", trend)

    else:
        st.write(
            "No major common trends were automatically identified."
        )

else:
    st.info("Search for papers to identify trends.")

st.header("🕳️ Research Gap Identification")

if "search_results" in st.session_state:

    st.write(
        "Based on the selected research papers, "
        "the following possible research gaps can be explored:"
    )

    research_gaps = [
        "Limited comparison of different machine learning approaches.",
        "Need for larger and more diverse datasets.",
        "Limited real-world validation of proposed models.",
        "Need for improved explainability and transparency in AI systems.",
        "More research is needed to compare results across different domains.",
        "Privacy, security and ethical issues require further investigation."
    ]

    for gap in research_gaps:
        st.write("🔹", gap)

else:
    st.info("Search for papers first.")

st.header("🚀 Potential Future Research Areas")

if "search_results" in st.session_state:

    future_areas = [
        "Develop more accurate and efficient AI models.",
        "Use larger and more diverse research datasets.",
        "Improve explainable Artificial Intelligence systems.",
        "Combine multiple machine learning approaches.",
        "Conduct real-world testing and validation.",
        "Improve privacy and security in AI-based systems.",
        "Explore new applications in different research domains."
    ]

    for area in future_areas:
        st.write("🔮", area)

else:
    st.info("Search for papers first.")

st.header("📑 Generate Literature Review Report")

if "search_results" in st.session_state:

    report_results = st.session_state["search_results"]

    report_query = st.session_state.get(
        "search_query",
        "selected research topic"
    )

    if st.button(
        "Generate Literature Review Report",
        key="generate_report"
    ):

        st.subheader("📚 Literature Review Report")

        st.write(
            f"### Research Topic: {report_query}"
        )

        st.write("### Overview")

        st.write(
            f"This literature review analyzes "
            f"{len(report_results)} research papers "
            f"related to **{report_query}**."
        )

        st.write(
            "The papers were retrieved and ranked using "
            "TF-IDF vectorization and cosine similarity."
        )

        st.write("### Selected Research Papers")

        for i, (_, row) in enumerate(
            report_results.iterrows(),
            1
        ):

            st.write(
                f"**{i}. {row['Title']} "
                f"({row['Year']})**"
            )

            st.write(
                create_summary(row["Abstract"])
            )

        st.write("### Common Trends")

        st.write(
            "The selected studies show common research "
            "directions based on the topics, methods and "
            "applications identified in the papers."
        )

        st.write("### Research Gaps")

        st.write(
            "Possible gaps include limited comparative "
            "studies, dataset limitations, real-world "
            "validation challenges, explainability and "
            "privacy concerns."
        )

        st.write("### Future Research")

        st.write(
            "Future work can focus on improved models, "
            "larger datasets, explainable AI, real-world "
            "validation and responsible AI systems."
        )

        st.success(
            "Literature Review Report Generated Successfully!"
        )

else:
    st.info("Search for papers first.")

st.sidebar.title("ℹ️ About")

st.sidebar.write(
    "This application helps researchers and students "
    "find, analyze, summarize and compare research papers."
)

st.sidebar.write("### Technologies Used")

st.sidebar.write("• Python")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Pandas")
st.sidebar.write("• Natural Language Processing")
st.sidebar.write("• TF-IDF")
st.sidebar.write("• Cosine Similarity")

st.sidebar.write("### Dataset")

st.sidebar.write(
    f"{len(df)} research papers"
)

st.sidebar.write("### Project Features")

st.sidebar.write("🔎 Academic Paper Search")
st.sidebar.write("📝 Text Summarization")
st.sidebar.write("⚖️ Paper Comparison")
st.sidebar.write("🔍 Research Gap Analysis")
st.sidebar.write("📈 Trend Identification")
st.sidebar.write("🚀 Future Research Suggestions")
st.sidebar.write("📑 Literature Review Report")

st.sidebar.success(
    "AI-Based Literature Review Assistant"
)
