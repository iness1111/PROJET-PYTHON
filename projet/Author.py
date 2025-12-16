# Author.py
from statistics import mean

class Author:
    """
    Classe représentant un auteur et les documents qu'il a produits.
    """

    def __init__(self, name):
        self.name = name              # nom de l'auteur
        self.ndoc = 0                 # nombre de documents
        self.production = {}          # dictionnaire {id_doc: Document}

    def add(self, id_doc, document):
        """
        Ajoute un document à la production de l'auteur.
        """
        self.production[id_doc] = document
        self.ndoc = len(self.production)  # met à jour le nombre de docs

    def taille_moyenne_docs(self):
        """
        Calcule la taille moyenne (en nombre de mots) des documents de l'auteur.
        """
        if not self.production:
            return 0
        tailles = [len(doc.texte.split()) for doc in self.production.values()]
        return mean(tailles)

    def __str__(self):
        """
        Représentation courte d’un auteur.
        """
        return f"Auteur : {self.name} ({self.ndoc} documents)"
