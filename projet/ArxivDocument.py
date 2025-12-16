from Document import Document

class ArxivDocument(Document):
    def __init__(self, titre, authors, date, url, texte):
        # authors is a list, Document wants a single string
        main_author = authors[0] if isinstance(authors, list) else authors
        
        super().__init__(titre, main_author, date, url, texte)

        self.coauthors = authors   # keep full list
        self._type = "arxiv"

    def __str__(self):
        return super().__str__() + f" | coauthors: {', '.join(self.coauthors)}"
