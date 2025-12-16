# RedditDocument.py
from Document import Document
import praw
import datetime

class RedditDocument(Document):
    def __init__(self, titre, auteur, date, url, texte, nb_comments):
        super().__init__(titre, auteur, date, url, texte)
        self._type = "reddit"
        self._nb_comments = nb_comments

    def __str__(self):
        return (
            f"[REDDIT] {self.titre} | {self.auteur} | {self.date}"
            f" | comments: {self._nb_comments}"
        )

    def getNbComments(self):
        return self._nb_comments

    def setNbComments(self, value):
        if not isinstance(value, int):
            raise ValueError("Le nombre de commentaires doit être un entier.")
        self._nb_comments = value

    @staticmethod
    def fetch_reddit_posts(client_id, client_secret, user_agent, subreddit_name, nbposts=10):
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
        subreddit = reddit.subreddit(subreddit_name)
        raw_docs = []
    
        for post in subreddit.hot(limit=nbposts):
            raw_docs.append({
                "titre": post.title,
                "auteur": str(post.author),
                "date": datetime.datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "url": post.url,
                "texte": post.selftext,
                "nb_comments": post.num_comments
            })
    
        return raw_docs
