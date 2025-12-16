from Document import Document

class RedditDocument(Document):
    def __init__(self, titre, auteur, date, url, texte, nb_comments):
        super().__init__(titre, auteur, date, url, texte)
        self._type = "reddit"
        self._nb_comments = nb_comments


    def __str__(self):
        return super().__str__() + f" | comments: {self.nb_comments}"

    def getNbComments(self):
            return self._nb_comments

    def setNbComments(self, value):
        # tu peux ajouter un contrôle si tu veux
        if not isinstance(value, int):
            raise ValueError("Le nombre de commentaires doit être un entier.")
        self._nb_comments = value