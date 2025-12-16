import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm


class SearchEngine:

    def __init__(self, corpus, stop_words=None, use_tfidf=True):
        """
        Initialize the search engine:
        - build vocab
        - build TF matrix
        - compute vocab stats (TF + DF)
        - build TF-IDF matrix
        """
        self.corpus = corpus
        self.stop_words = stop_words if stop_words else set()
        self.use_tfidf = use_tfidf

        # 1️⃣ Build vocabulary
        self.vocab = corpus.build_vocab_dict(stop_words=self.stop_words)

        # 2️⃣ Build TF matrix
        self.mat_TF, self.vocab = corpus.build_TF_matrix(stop_words=self.stop_words)

        # 3️⃣ Update vocabulary stats
        self.vocab = corpus.update_vocab_stats(self.mat_TF, self.vocab)

        # 4️⃣ Build TF-IDF matrix
        self.mat_TF_IDF = corpus.build_TF_IDF_matrix(self.mat_TF, self.vocab)

        # Create fast mapping word → column index in matrix
        self.word_to_col = {w: info["id"] - 1 for w, info in self.vocab.items()}

        print("SearchEngine initialized:")
        print(f"- vocab size: {len(self.vocab)}")
        print(f"- matrix shape: {self.mat_TF.shape}")

    # -------------------------------------------------------
    # SEARCH FUNCTION (MAIN PART 3 REQUIREMENT)
    # -------------------------------------------------------
    def search(self, query, k=10):  
        """
        Search documents for given query terms.
        Returns a pandas DataFrame of the top-k documents.
        """
        query = self.corpus.nettoyer_texte(query)
        tokens = query.split()
        tokens = [w for w in tokens if w in self.vocab]

        if not tokens:
            return pd.DataFrame({"message": ["No query terms found in vocabulary"]})

        # Build query vector
        q_vec = np.zeros(len(self.vocab))
        for w in tokens:
            q_vec[self.word_to_col[w]] += 1 # the column that corresponds to each word := 1
        q_vec = csr_matrix(q_vec)

        # Choose matrix
        doc_matrix = self.mat_TF_IDF if self.use_tfidf else self.mat_TF

        # Cosine similarity
        sims = cosine_similarity(q_vec, doc_matrix).flatten()

        # Top-k ranking
        top_idx = sims.argsort()[::-1][:k]

        # Build DataFrame with progress bar
        rows = []
        for idx in tqdm(top_idx, desc="Processing top-k documents", unit="doc"):
            doc_id = idx + 1
            doc = self.corpus.id2doc[doc_id]

            # 🔥 CALL CONCORD INSIDE SEARCH ENGINE
            concord_text = self.corpus.concord(
                doc_id,
                query_words=tokens,  # concord only handles a single word
                window=50
            )
        
            rows.append({
                "doc_id": doc_id,
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date,
                "score": sims[idx],
                "url": getattr(doc, "url", ""),
                "texte": concord_text
            })

        return pd.DataFrame(rows)
