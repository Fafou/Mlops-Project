"""Nettoyage du texte des commentaires."""

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9àâäçéèêëîïôöùûüÿœ' ]+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Met en minuscules, retire les URL et la ponctuation, normalise les espaces."""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
