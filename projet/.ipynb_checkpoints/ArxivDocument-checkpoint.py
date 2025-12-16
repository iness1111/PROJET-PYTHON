# ArxivDocument.py
from Document import Document
import urllib.request as libreq
import xmltodict

class ArxivDocument(Document):
    def __init__(self, titre, authors, date, url, texte):
        main_author = authors[0] if isinstance(authors, list) else authors
        super().__init__(titre, main_author, date, url, texte)

        self.coauthors = authors
        self._type = "arxiv"

    def __str__(self):
        return (
            f"[ARXIV] {self.titre} | {self.auteur} | {self.date}"
            f" | coauthors: {', '.join(self.coauthors)}"
        )


    
    @staticmethod
    def fetch_arxiv(theme="machine+learning", max_results=10):
        url = f"http://export.arxiv.org/api/query?search_query=all:{theme}&start=0&max_results={max_results}"
        req = libreq.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = libreq.urlopen(req).read()
        feed = xmltodict.parse(data)
    
        entries = feed["feed"].get("entry", [])
        if not isinstance(entries, list):
            entries = [entries]
    
        raw_docs = []
    
        for entry in entries:
            authors = (
                [a["name"] for a in entry["author"]]
                if isinstance(entry["author"], list)
                else [entry["author"]["name"]]
            )
    
            url_link = ""
            for l in entry.get("link", []):
                if l.get("@rel") == "alternate":
                    url_link = l.get("@href", "")
    
            raw_docs.append({
                "titre": entry["title"].strip(),
                "auteur": authors[0],
                "date": entry["published"],
                "url": url_link,
                "texte": entry["summary"]
            })
    
        return raw_docs
