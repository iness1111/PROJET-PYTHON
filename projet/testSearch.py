import unittest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from SearchEngine import SearchEngine
from Corpus import Corpus
from Document import Document

# -----------------------------
# Mock corpus for testing
# -----------------------------
class MockCorpus(Corpus):
    def __init__(self):
        super().__init__("MockCorpus")
        # Add documents using factory — no 'id' argument!
        self.add_document(Corpus.factory(
            "Machine learning is great",  # texte
            "Doc 1",                      # titre
            "Alice",                      # auteur
            "2025-01-01",                 # date
            "csv"                         # source
        ))
        self.add_document(Corpus.factory(
            "Deep learning applications",
            "Doc 2",
            "Bob",  
            "2025-01-02",
            "csv"
        ))
        self.add_document(Corpus.factory(
            "Data science and AI",
            "Doc 3",
            "Alice",
            "2025-01-03",
            "csv"
        ))

        
    def nettoyer_texte(self, texte):
        return texte.lower()
    
    def concord(self, doc_id, query_words, window=50):
        # simply return the text for testing
        return self.id2doc[doc_id].texte

# -----------------------------
# Test class
# -----------------------------
class TestSearchEngine(unittest.TestCase):

    def setUp(self):
        corpus = MockCorpus()
        self.engine = SearchEngine(corpus, use_tfidf=False)
        # override vocab for predictable test
        self.engine.vocab = {"machine":0, "learning":1, "deep":2, "data":3, "science":4, "ai":5, "applications":6}
        self.engine.word_to_col = {w:i for i,w in enumerate(self.engine.vocab)}
        self.engine.mat_TF = csr_matrix(np.array([
            [1,1,0,0,0,0,0],  # Doc 1
            [0,1,1,0,0,0,1],  # Doc 2
            [0,0,0,1,1,1,0],  # Doc 3
        ]))
        self.engine.mat_TF_IDF = self.engine.mat_TF  # simple for test

    def test_search_returns_dataframe(self):
        results = self.engine.search("machine")
        self.assertIsInstance(results, pd.DataFrame)

    def test_search_top_k(self):
        results = self.engine.search("learning", k=2)
        self.assertEqual(len(results), 2)

    def test_search_no_vocab_terms(self):
        results = self.engine.search("quantum")
        self.assertIn("message", results.columns)
        self.assertEqual(results["message"].iloc[0], "No query terms found in vocabulary")

    def test_search_doc_fields(self):
        results = self.engine.search("data")
        first_row = results.iloc[0]
        self.assertIn("doc_id", first_row)
        self.assertIn("titre", first_row)
        self.assertIn("auteur", first_row)
        self.assertIn("texte", first_row)
        self.assertIn("score", first_row)

if __name__ == "__main__":
    unittest.main()
