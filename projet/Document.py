# Document.py
from datetime import datetime

class Document:
    """
    Classe représentant un document textuel avec ses métadonnées.
    """

    def __init__(self, titre, auteur, date, url, texte):
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte
        self._type = "document" # val par defaut


    def afficher(self):
        """Affiche toutes les informations du document."""
        print(f"Titre : {self.titre}")
        print(f"Auteur : {self.auteur}")
        print(f"Date : {self.date}")
        print(f"URL : {self.url}")
        print(f"Texte (début) : {self.texte[:200]}...")  # seulement les 200 premiers caractères
        print(f"Type : {self._type}")

    def __str__(self):
        """Renvoie une version courte et lisible de l'objet."""
        return f"[{self._type}] {self.titre} — {self.auteur} ({self.date})"

    def getType(self):
        return self._type


    @staticmethod
    def fetch_csv(
        file_path,
        texte_col="Texte",
        titre_col="Titre",
        auteur_col="Auteur",
        date_col="Date",
        url_col="URL"
    ):
        """
        Generic CSV fetcher: returns raw rows as dicts
        (NOT Document objects)
        """
        df = pd.read_csv(file_path)
        return df