"""Tests unitaires du nettoyage de texte."""

from toxiguard.data.preprocess import clean_text


def test_minuscules_et_espaces():
    assert clean_text("  Hello   WORLD ") == "hello world"


def test_retire_les_urls():
    assert clean_text("voir https://example.com maintenant") == "voir maintenant"


def test_retire_la_ponctuation_et_garde_les_accents():
    assert clean_text("Nul !!! Dégage...") == "nul dégage"


def test_chaine_vide():
    assert clean_text("") == ""
