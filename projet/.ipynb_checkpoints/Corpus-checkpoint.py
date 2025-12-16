# Corpus.py
import pandas as pd
import pickle
from Document import Document
from ArxivDocument import ArxivDocument
from RedditDocument import RedditDocument
from Author import Author
import csv  

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

    def __repr__(self):
        return f"<Corpus '{self.nom}' — {self.ndoc} docs / {self.naut} auteurs>"

    
    # new one (clean save)
    def save(self, filename):
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



    
    @staticmethod
    def load(filename, name="LoadedCorpus"):
        """
        Charge un corpus à partir d’un fichier TSV.
        """
        df = pd.read_csv(filename, sep="\t", encoding="utf-8")
        corpus = Corpus(name)
        for _, row in df.iterrows():
            doc_type = row["type"]
            if doc_type == "arxiv":
                doc = ArxivDocument(row["titre"], row["auteur"], row["date"], row["texte"], ...)
            elif doc_type == "reddit":
                doc = RedditDocument(row["titre"], row["auteur"], row["date"], row["texte"], ...)
            else:
                doc = Document(row["titre"], row["auteur"], row["date"], row["texte"])
            corpus.add_document(doc)
        print(f"Corpus loaded from {filename} ({len(df)} documents).")
        return corpus

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
            return ArxivDocument(titre, auteur, date, texte, url)
        
        # fallback générique
        return Document(titre, auteur, date, texte)


    def build_big_string(self):
        """
        Concatène tous les textes (lazy : construit une seule fois).
        """
        if self._big_string is None:
            self._big_string = " ".join([doc.texte for doc in self.id2doc.values()])
        return self._big_string

    def search(self, motif):
        """
        Retourne toutes les occurrences du motif.
        """
        big = self.build_big_string()
        pattern = re.compile(motif, re.IGNORECASE)
        return pattern.findall(big)









