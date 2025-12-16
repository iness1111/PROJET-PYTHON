import streamlit as st
import pandas as pd
from SearchEngine import SearchEngine
from Corpus import Corpus
from nltk.corpus import stopwords

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Discours Search Engine",
    layout="wide"
)
st.title("📚 Discours Search Engine")
st.markdown("Search through political speeches using keywords.")

stop_words = set(stopwords.words('english'))

# --------------------------------------------------
# 2. LOAD CORPUS & SEARCH ENGINE (cached)
# --------------------------------------------------
@st.cache_resource
def load_engine():
    corpus = Corpus.load("DiscoursCorpus.csv", name="Search_Discours")
    engine = SearchEngine(
        corpus,
        stop_words=stop_words,
        use_tfidf=True
    )
    return engine

engine = load_engine()
corpus = engine.corpus  # for filters

# --------------------------------------------------
# 3. SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("🔍 Search Settings")

query = st.sidebar.text_input(
    "Enter keywords",
    placeholder="election, democracy, Americans..."
)

k = st.sidebar.slider(
    "Number of results",
    min_value=1,
    max_value=50,
    value=10
)

# --- Filters ---
authors = sorted(set(doc.auteur for doc in corpus.id2doc.values()))
selected_author = st.sidebar.selectbox("Filter by author (optional)", ["All"] + authors)

dates = sorted(set(doc.date for doc in corpus.id2doc.values()))
selected_date = st.sidebar.selectbox("Filter by date (optional)", ["All"] + dates)

search_btn = st.sidebar.button("Search")

# --------------------------------------------------
# 4. OUTPUT AREA
# --------------------------------------------------
output_container = st.container()

# --------------------------------------------------
# 5. SEARCH ACTION
# --------------------------------------------------
if search_btn:
    if query.strip() == "":
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching documents..."):
            results = engine.search(query, k=500)  # get more results to apply filters

            # Apply filters
            if selected_author != "All":
                results = results[results["auteur"] == selected_author]
            if selected_date != "All":
                results = results[results["date"] == selected_date]

            results = results.head(k)  # limit to selected number

        with output_container:
            if "message" in results.columns:
                st.info(results["message"].iloc[0])
            else:
                st.success(f"Top {len(results)} results after filters")

                st.dataframe(
                    results[["doc_id", "titre", "auteur", "score"]],
                    use_container_width=True
                )

                # Expandable detailed view
                st.markdown("### 📄 Document previews")
                for _, row in results.iterrows():
                    with st.expander(f"{row['titre']} (score: {row['score']:.3f})"):
                        st.markdown(f"**Author:** {row['auteur']}")
                        st.markdown(f"**Date:** {row['date']}")
                        st.markdown(row["texte"])
