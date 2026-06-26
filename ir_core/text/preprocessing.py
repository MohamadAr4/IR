"""Canonical text preprocessing, shared by documents AND queries.

Requirement #4 ("Query Processing") states that queries must be processed with
*the same* techniques as the documents so the two live in the same term space.
The preprocessed corpus JSON files were built with this exact pipeline
(lowercase -> strip punctuation -> tokenize -> drop stopwords -> Porter stem),
so :func:`process` reproduces it and is the single source of truth for both
sides of retrieval.
"""
from __future__ import annotations

import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# --- make sure the NLTK data we rely on is present (download once) ----------
_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
for _name, _path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_name, quiet=True)
        except Exception:
            pass

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
try:
    _STOPWORDS_EN = set(stopwords.words("english"))
except Exception:
    _STOPWORDS_EN = set()

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _wordnet_pos(treebank_tag: str):
    if not treebank_tag:
        return wordnet.NOUN
    return {"J": wordnet.ADJ, "N": wordnet.NOUN,
            "V": wordnet.VERB, "R": wordnet.ADV}.get(treebank_tag[0].upper(), wordnet.NOUN)


def process(text: str,
            *,
            lowercase: bool = True,
            remove_punctuation: bool = True,
            remove_stopwords: bool = True,
            do_lemmatize: bool = False,
            do_stemming: bool = True) -> list[str]:
    """Return the list of normalized tokens for ``text``.

    Defaults match how the corpus JSON was generated (stemming on). The same
    function is called for queries, guaranteeing term-space agreement.
    """
    if not text:
        return []

    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(_PUNCT_TABLE)

    tokens = [t for t in word_tokenize(text) if t.isalpha()]

    if remove_stopwords and _STOPWORDS_EN:
        tokens = [t for t in tokens if t not in _STOPWORDS_EN]

    if do_lemmatize:
        try:
            tokens = [_LEMMATIZER.lemmatize(w, _wordnet_pos(tag)) for w, tag in pos_tag(tokens)]
        except Exception:
            pass

    if do_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


def stem(term: str) -> str:
    """Stem a single term (used by query-expansion so synonyms land in the
    same stemmed term space as the index)."""
    return _STEMMER.stem(term.lower())
