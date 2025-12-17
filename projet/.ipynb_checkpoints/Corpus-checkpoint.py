# Corpus.py
import pandas as pd
import pickle
from Document import Document
from ArxivDocument import ArxivDocument
from RedditDocument import RedditDocument
from Author import Author
import csv  
import re
from collections import Counter
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Corpus:
    _instance = None  # Singleton TD5P4
    
    def __new__(cls, *args, **kwargs): # TD5P4 garantit qu’un seul objet Corpus peut exister
        if cls._instance is None:
            cls._instance = super(Corpus, cls).__new__(cls)
        return cls._instance

    def __init__(self, nom):
        self.nom = nom # le nom du corpus
        self.id2doc = {} # le dictionnaire des documents
        self.authors = {} # le dictionnaire des auteurs
        self.ndoc = 0 # comptage des documents
        self.naut = 0 #  comptage des auteurs
        self._full_text = None # text concatené pour ne pas reconstruire la chaine a chaque search

        self.vocab = {}          # dictionary {word: {"id":..., "total_occurrences":..., "document_frequency":...}}
        self.mat_TF = None       # sparse CSR matrix of TF
        self.mat_TFXIDF = None   # sparse CSR matrix of TF-IDF


    # -----------------------------
    #  MÉTHODES D’AJOUT
    # -----------------------------
    def add_document(self, document):
        """
        Ajoute un document au corpus et met à jour les auteurs.
        """
        doc_id = self.ndoc + 1
        self.id2doc[doc_id] = document
        self.ndoc += 1
        self._full_text = None  # reset full_text cache when a new doc is added

        # Gestion de l’auteur
        auteur_nom = document.auteur
        if auteur_nom not in self.authors:
            self.authors[auteur_nom] = Author(auteur_nom)
        self.authors[auteur_nom].add(doc_id, document)
        self.naut = len(self.authors)

    # -----------------------------
    #  AFFICHAGE
    # -----------------------------
    def show_by_date(self, n=5):
        """
        Affiche les n documents les plus récents.
        """
        docs_sorted = sorted(
            self.id2doc.values(),
            key=lambda d: d.date,
            reverse=True
        )
        for doc in docs_sorted[:n]:
            print(f"{doc.date} — {doc.titre} ({doc.auteur})")

    def show_by_title(self, n=5):
        """
        Affiche les n premiers documents triés par titre.
        """
        docs_sorted = sorted(
            self.id2doc.values(),
            key=lambda d: d.titre.lower()
        )
        for doc in docs_sorted[:n]:
            print(f"{doc.titre} — {doc.auteur}")

    def __repr__(self): # info displayed for the csv saving
        return f"<Corpus '{self.nom}' — {self.ndoc} docs / {self.naut} auteurs>"

    
    # new one (clean save)
    def save(self, filename, min_length=20, clean=True):
        if clean:
            self.remove_small_docs(min_length=min_length)
    
        data = []
        for doc_id, doc in self.id2doc.items():
            data.append({
                "id": doc_id,
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date,
                "url": getattr(doc, "url", ""),
                "texte": " ".join(doc.texte.split()),
                "type": doc.getType()
            })
    
        df = pd.DataFrame(data)
        df.to_csv(filename, sep=';', index=False, encoding="utf-8")
        return df



    @staticmethod
    def load(filename, name="LoadedCorpus"):
        import pandas as pd
        df = pd.read_csv(filename, sep=";", encoding="utf-8")
        corpus = Corpus(name)
    
        for _, row in df.iterrows():
            # fallback to generic document if 'type' missing
            doc_type = row.get("type", "csv")
            texte = row["texte"]
            titre = row["titre"]
            auteur = row["auteur"]
            date = row["date"]
            url = row.get("url", "")
            nb_comments = int(row.get("nb_comments", 0))
    
            # Use factory to create the right document type
            doc = Corpus.factory(
                texte=texte,
                titre=titre,
                auteur=auteur,
                date=date,
                source=doc_type,
                url=url,
                nb_comments=nb_comments
            )
            corpus.add_document(doc)
    
        print(f"Corpus loaded from {filename} ({len(corpus.id2doc)} documents).")
        return corpus

    

    def doc_stats(self):
        """
        For each document, print the number of words and sentences.
        Simplified: words split by spaces, sentences split by periods.
        """
        for doc_id, doc in self.id2doc.items():
            text = str(doc.texte)
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            print(f"Doc {doc_id}: '{doc.titre}' — Words: {word_count}, Sentences: {sentence_count}")

    def remove_small_docs(self, min_length=20):
        """
        Remove documents whose text is smaller than min_length characters.
        """
        to_remove = [doc_id for doc_id, doc in self.id2doc.items() if len(str(doc.texte)) < min_length]
        for doc_id in to_remove:
            removed_doc = self.id2doc.pop(doc_id)
            print(f"Removed Doc {doc_id}: '{removed_doc.titre}' ({len(str(removed_doc.texte))} chars)")
        self.update_counts() 

    def update_counts(self):
        self.ndoc = len(self.id2doc)
        self.naut = len(self.authors)
 
    def get_full_text(self):
        """
        Return a single string containing all documents' texts joined together.
        Useful for TD 6 Partie 1 and 2.
        """
        
        full_text = []
        for doc in self.id2doc.values():   # make sure self.documents is not empty!
            try:
                full_text.append(str(doc.texte))
            except Exception as e:
                print("Error reading doc:", e)
        self._full_text = "\n".join(full_text)
        return self._full_text


    # FACTORY PATTERN TD5P4
    @staticmethod 
    def factory(texte, titre, auteur, date, source, url="", nb_comments=0):
        """
        Factory : crée automatiquement le bon type de document selon la source.
        """
        texte = " ".join(texte.split())  # nettoyage minimal
        
        if source == "reddit":
            return RedditDocument(titre, auteur, date, url, texte, nb_comments)
        
        if source == "arxiv":
            return ArxivDocument(titre, auteur, date, url, texte)

        if source == "csv":
            doc = Document(titre, auteur, date, url, texte)
            doc._type = "csv"
            return doc
        
        
        # fallback générique
        return Document(titre, auteur, date, url, texte)


    
    def concord(self, doc_id, query_words, window=50):
        """
        Returns a snippet from the document containing the query words.
        Highlights query words in bold (**word**).
        query_words: list of words (or a single word as string)
        window: number of words before and after the first match
        """
        if isinstance(query_words, str):
            query_words = [query_words]
    
        doc = self.id2doc[doc_id]
        words = self.nettoyer_texte(doc.texte).split()
    
        # Find all indices where any query word appears
        indices = [i for i, w in enumerate(words) if w in query_words]
        if not indices:
            return ""  # none of the query words found
    
        # Take first match
        idx = indices[0]
    
        start = max(0, idx - window)
        end = min(len(words), idx + window + 1)
    
        snippet = words[start:end]
    
        # Highlight all query words
        snippet = [f"**{w}**" if w in query_words else w for w in snippet]
    
        return " ".join(snippet)


    


     # ----------------------------
    # Part 2.1: Text cleaning
    # ----------------------------
    @staticmethod
    def nettoyer_texte(text):
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"\d+", "", text)            # remove digits
        text = re.sub(r"[^\w\s]", "", text)        # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()   # remove extra spaces
        return text

    # ----------------------------
    # Part 2.2-2.4: Statistics
    # ----------------------------
    # stop_words = set(stopwords.words('english'))  # set of stopwords for fast lookup

    def get_vocab(self, stop_words=None):
        """
        Returns the unique vocabulary of the corpus, sorted alphabetically.
        """
        if stop_words is None:
            stop_words = set()
        
        vocab = set()
        for doc in self.id2doc.values():
            cleaned_text = self.nettoyer_texte(doc.texte)
            words = cleaned_text.split()
            words = [w for w in words if w not in stop_words]
            vocab.update(words)
        
        return sorted(vocab)

    

    # ----------------------------
    # 1.1 Build vocabulary
    # ----------------------------
    def build_vocab_dict(self, stop_words=None):
        """
        Builds a vocabulary dictionary using get_vocab().
        Keys = words
        Values = {
            'id': unique ID,
            'total_occurrences': 0,
            'document_frequency': 0
        }
        """
        if stop_words is None:
            stop_words = set()
    
        # 1️⃣ Get sorted unique vocab
        vocab_list = self.get_vocab(stop_words=stop_words)
    
        # 2️⃣ Initialize dictionary
        vocab_dict = {
            w: {
                "id": idx + 1,
                "total_occurrences": 0,
                "document_frequency": 0
            }
            for idx, w in enumerate(vocab_list)
        }
    
        return vocab_dict


    # ----------------------------
    # 1.2 Build TF matrix
    # ----------------------------
    def build_TF_matrix(self, stop_words=None):   # Each row = a document, each column = a word
        vocab_dict = self.build_vocab_dict(stop_words)
        n_docs = len(self.id2doc)
        n_words = len(vocab_dict)

        data, rows, cols = [], [], []
        word_to_col = {w: vocab_dict[w]["id"] - 1 for w in vocab_dict}

        for row_idx, doc in enumerate(self.id2doc.values()):
            words = self.nettoyer_texte(doc.texte).split()
            counts = Counter(w for w in words if w in vocab_dict)
            for w, count in counts.items():
                col_idx = word_to_col[w]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(count)

        mat_TF = csr_matrix((data, (rows, cols)), shape=(n_docs, n_words))
        return mat_TF, vocab_dict

    # ----------------------------
    # 1.3 Update vocab stats
    # ----------------------------
    @staticmethod  
    def update_vocab_stats(mat_TF, vocab_dict):
        n_words = mat_TF.shape[1]
        for word, info in vocab_dict.items():
            col_idx = info["id"] - 1
            col = mat_TF[:, col_idx].toarray().flatten()
            info["total_occurrences"] = int(col.sum())
            info["document_frequency"] = int(np.count_nonzero(col))
        return vocab_dict

    # ----------------------------
    # 1.4 Build TF-IDF matrix
    # ----------------------------
    @staticmethod
    def build_TF_IDF_matrix(mat_TF, vocab_dict): 
        N = mat_TF.shape[0]  # number of documents
        n_words = mat_TF.shape[1]
        idf = np.zeros(n_words)
        for word, info in vocab_dict.items():
            col_idx = info["id"] - 1
            df = info["document_frequency"]
            idf[col_idx] = np.log((N + 1) / (df + 1)) + 1  # smooth IDF

        mat_TF_IDF = mat_TF.multiply(idf)
        return mat_TF_IDF
