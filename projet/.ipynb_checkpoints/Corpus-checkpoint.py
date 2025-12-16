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
    def save(self, filename, min_length=20):
        self.remove_small_docs(min_length=min_length)
        data = []
        for doc_id, doc in self.id2doc.items():
            data.append({
                "id": doc_id,
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date,
                "url": getattr(doc, "url", ""),  # safe for RedditDocument
                "texte": " ".join(doc.texte.split()),  # remove newlines
                # "nb_comments": doc.nb_comments,  # 0 if not available
                "type": doc.getType()
            })
    
        df = pd.DataFrame(data)
        df.to_csv(filename, sep=';', index=False, encoding="utf-8")
        print(f"Corpus saved to {filename} ({len(df)} documents).")



    
    # @staticmethod
    # def load(filename, name="LoadedCorpus"):
    #     """
    #     Charge un corpus à partir d’un fichier TSV.
    #     """
    #     df = pd.read_csv(filename, sep=";", encoding="utf-8")
    #     corpus = Corpus(name)
    #     for _, row in df.iterrows():
    #         doc_type = row["type"]
    #         if doc_type == "arxiv":
    #             doc = ArxivDocument(row["titre"], row["auteur"], row["date"], row["texte"], ...)
    #         elif doc_type == "reddit":
    #             doc = RedditDocument(row["titre"], row["auteur"], row["date"], row["texte"], ...)
    #         else:
    #             doc = Document(row["titre"], row["auteur"], row["date"], row["texte"])
    #         corpus.add_document(doc)
    #     print(f"Corpus loaded from {filename} ({len(df)} documents).")
    #     return corpus

    @staticmethod
    def load(filename, name="LoadedCorpus"):
        import pandas as pd
        df = pd.read_csv(filename, sep=";", encoding="utf-8")
        corpus = Corpus(name)
    
        for _, row in df.iterrows():
            doc_type = row["type"]
            texte = row["texte"]
            titre = row["titre"]
            auteur = row["auteur"]
            date = row["date"]
            url = row.get("url", "")
            nb_comments = int(row.get("nb_comments", 0))  # safe fallback for Reddit
    
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
    
        print(f"Corpus loaded from {filename} ({len(df)} documents).")
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
        # full_text = " ".join(str(doc.texte) for doc in self.id2doc.values())
        # return full_text

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
        
        # fallback générique
        return Document(titre, auteur, date, texte)


    

    def search(self, keyword):
        """
        Returns a list of document texts that contain the given keyword.
        Case-insensitive search.
        """
        if self._full_text is None:
            # build full_text once
            self._full_text = self.get_full_text()
        
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
        for doc_id, doc in self.id2doc.items():
            if pattern.search(str(doc.texte)):  # check if keyword is in the document text
                results.append({
                    "id": doc_id,
                    "auteur": doc.auteur,
                    "titre": doc.titre,
                    "url": getattr(doc, "url", ""),
                    "texte": doc.texte
                })

        return results
    
    # def concorde(self, keyword, context_size=30):
    #     """
    #     Returns a list of document texts that contain the given keyword showing left and right context around the match
    #     Case-insensitive search.
    #     """
    #     if self._full_text is None:
    #         # build full_text once
    #         self._full_text = self.get_full_text()
        
    #     results = []
    #     pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
    #     for doc_id, doc in self.id2doc.items():
    #         text = str(doc.texte)
    #         for match in pattern.finditer(text):
    #             start, end = match.span()
    #             concord_text = text[max(0, start - context_size):end + context_size]
    #             results.append({
    #                 "id": doc_id,
    #                 "auteur": doc.auteur,
    #                 "titre": doc.titre,
    #                 "url": getattr(doc, "url", ""),
    #                 "texte": concord_text
    #             })

    #     df = pd.DataFrame(results)
    #     return df


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

    # ----------------------------
    # Convenience: get top stats
    # ----------------------------
    # def stats(self, n=10, stop_words=None):
    #     mat_TF, vocab_dict = self.build_TF_matrix(stop_words)
    #     vocab_dict = self.update_vocab_stats(mat_TF, vocab_dict)

    #     words, tf, df = [], [], []
    #     for word, info in vocab_dict.items():
    #         words.append(word)
    #         tf.append(info["total_occurrences"])
    #         df.append(info["document_frequency"])

    #     freq_df = pd.DataFrame({
    #         "word": words,
    #         "term_frequency": tf,
    #         "document_frequency": df
    #     })

    #     return freq_df.sort_values(by="term_frequency", ascending=False).head(n)




    # def search_engine(self, query, top_k=10, use_tfidf=True, stop_words=None, debug=False):
    #     """
    #     Robust search: auto-builds vocab/matrices if missing.
    #     Returns top_k documents ranked by cosine similarity.
    #     """
    #     if stop_words is None:
    #         stop_words = set()
    
    #     # --- ensure vocab & TF matrix exist ---
    #     if not self.vocab:  # empty dict or not built
    #         if debug: print("Building vocab_dict via build_vocab_dict()...")
    #         self.vocab = self.build_vocab_dict(stop_words=stop_words)
    
    #     if self.mat_TF is None:
    #         if debug: print("Building TF matrix via build_TF_matrix()...")
    #         mat_TF, vocab_dict = self.build_TF_matrix(stop_words=stop_words)
    #         # update instance attributes so later calls use them
    #         self.mat_TF = mat_TF
    #         # update self.vocab with returned vocab_dict stats (ids)
    #         # build_TF_matrix returns vocab_dict local — keep it in self.vocab shape
    #         self.vocab = vocab_dict
    
    #         # IMPORTANT: build_TF_matrix as written does NOT update total_occurrences/document_frequency.
    #         # If you want those stats, call update_vocab_stats (you have it as static).
    #         try:
    #             self.vocab = self.update_vocab_stats(self.mat_TF, self.vocab)
    #         except Exception:
    #             pass
    
    #     # build TF-IDF if missing
    #     if self.mat_TFXIDF is None:
    #         if debug: print("Building TF-IDF matrix via build_TF_IDF_matrix()...")
    #         # your build_TF_IDF_matrix is a @staticmethod(mat_TF, vocab_dict)
    #         try:
    #             self.mat_TFXIDF = self.build_TF_IDF_matrix(self.mat_TF, self.vocab)
    #         except TypeError:
    #             # fallback: if method is instance-style without args
    #             self.mat_TFXIDF = self.build_TF_IDF_matrix()
    
    #     # --- clean query and tokens ---
    #     query_clean = self.nettoyer_texte(query)
    #     tokens = [t for t in query_clean.split() if t and (t not in stop_words)]
    #     # keep only tokens present in vocab
    #     query_words = [t for t in tokens if t in self.vocab]
    
    #     if debug:
    #         print("query_clean:", query_clean)
    #         print("tokens:", tokens)
    #         print("tokens in vocab:", query_words)
    
    #     if not query_words:
    #         if debug:
    #             print("No valid query terms found in vocabulary.")
    #             print("Try: (1) rebuild vocab with same stop_words, (2) check tokenization or plural/sing forms.")
    #         return []
    
    #     # --- build query vector aligned with vocab IDs ---
    #     import numpy as np
    #     from scipy.sparse import csr_matrix
    #     V = len(self.vocab)
    #     vec = np.zeros(V, dtype=float)
    #     for w in query_words:
    #         idx = self.vocab[w]['id'] - 1
    #         vec[idx] += 1.0
    #     query_vec = csr_matrix(vec.reshape(1, -1))
    
    #     # --- choose document matrix ---
    #     doc_matrix = self.mat_TFXIDF if use_tfidf else self.mat_TF
    
    #     # --- compute cosine similarities ---
    #     sims = cosine_similarity(query_vec, doc_matrix).flatten()
    
    #     # --- map matrix row indices to actual doc_id keys ---
    #     doc_id_list = list(self.id2doc.keys())  # ordering must match build_TF_matrix enumeration
    #     # If build_TF_matrix used enumerate(self.id2doc.values()), this mapping is consistent.
    
    #     # get top indices with positive score
    #     ranked = sims.argsort()[::-1]
    #     top = [i for i in ranked if sims[i] > 0][:top_k]  # ignore zero-score docs
    
    #     results = []
    #     for row_idx in top:
    #         # map row index to doc_id
    #         try:
    #             doc_id = doc_id_list[row_idx]
    #         except IndexError:
    #             # fallback: assume doc ids are 1-based contiguous
    #             doc_id = row_idx + 1
    #         doc = self.id2doc[doc_id]
    #         results.append({
    #             "id": doc_id,
    #             "titre": doc.titre,
    #             "auteur": doc.auteur,
    #             "url": getattr(doc, "url", ""),
    #             "score": float(sims[row_idx]),
    #             "texte": doc.texte
    #         })
    
    #     return results


        
    # def build_all(self, stop_words=None):
    #     if stop_words is None:
    #         stop_words = set()
    #     self.vocab = self.build_vocab_dict(stop_words=stop_words)
    #     self.mat_TF, self.vocab = self.build_TF_matrix(stop_words=stop_words)
    #     self.vocab = self.update_vocab_stats(self.mat_TF, self.vocab)
    #     self.mat_TFXIDF = self.build_TF_IDF_matrix(self.mat_TF, self.vocab)