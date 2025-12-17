import unittest
import os
from Corpus import Corpus
from testSearch import MockCorpus  # reuse mock

class TestCorpusIO(unittest.TestCase):

    def setUp(self):
        self.filename = "test_corpus.csv"
        self.corpus = MockCorpus()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_save(self):
        df = self.corpus.save(self.filename, clean=False)
        self.assertEqual(len(df), 3)
        self.assertTrue(os.path.exists(self.filename))

    def test_load(self):
        self.corpus.save(self.filename, clean=False)
        loaded = Corpus.load(self.filename)

        self.assertEqual(len(loaded.id2doc), 3)
        self.assertEqual(loaded.id2doc[1].titre, "Doc 1")
if __name__ == "__main__":
    unittest.main()
