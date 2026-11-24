"""Modèle de base : TF-IDF + régression logistique (une par étiquette)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxiguard.config import RANDOM_STATE


def build_pipeline(max_features: int = 50_000, c: float = 4.0) -> Pipeline:
    """Construit le pipeline : vectorisation TF-IDF (mots et bigrammes) puis classifieur multi-label."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    classifier = LogisticRegression(
        C=c,
        class_weight="balanced",  # classes très déséquilibrées
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", OneVsRestClassifier(classifier))])
