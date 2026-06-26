import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure required NLTK resources are available
_NLTK_RESOURCES = {
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4'
}

for name, path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(name)
        except Exception:
            pass

# Prepare common tools
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS_EN = set(stopwords.words('english'))


def _get_wordnet_pos(treebank_tag):
    if not treebank_tag:
        return wordnet.NOUN
    tag = treebank_tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text,
                    language='english',
                    lowercase=True,
                    remove_punctuation=True,
                    remove_stopwords=True,
                    do_stemming=False,
                    do_lemmatize=False,
                    do_pos_tag=False,
                    do_spell_check=False,
                    use_spacy=False):
    """Flexible preprocessing pipeline inspired by the provided notebook.

    Returns either a list of tokens or a dict with additional info when POS tagging is enabled.
    """
    if not text:
        return []

    # Lowercase
    if lowercase:
        text_proc = text.lower()
    else:
        text_proc = text

    # Optionally remove punctuation
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        text_proc = text_proc.translate(translator)

    # Tokenize
    tokens = word_tokenize(text_proc)

    # Filter tokens (keep alphabetic tokens)
    tokens = [t for t in tokens if t.isalpha()]

    # Stopwords
    if remove_stopwords and language.lower().startswith('en'):
        tokens = [t for t in tokens if t not in _STOPWORDS_EN]

    # POS tagging (if requested, used also for lemmatization)
    pos_tags = None
    if do_pos_tag or do_lemmatize:
        try:
            pos_tags = pos_tag(tokens)
        except Exception:
            pos_tags = [(t, None) for t in tokens]

    # Lemmatization
    if do_lemmatize:
        lemmatized = []
        for word, tag in pos_tags:
            wn_tag = _get_wordnet_pos(tag)
            lemmatized.append(_LEMMATIZER.lemmatize(word, wn_tag))
        tokens = lemmatized

    # Stemming (after lemmatization or on original tokens)
    if do_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    # Spell checking (optional, requires pyspellchecker)
    if do_spell_check:
        try:
            from spellchecker import SpellChecker

            spell = SpellChecker()
            corrected = []
            misspelled = spell.unknown(tokens)
            for t in tokens:
                if t in misspelled:
                    c = spell.correction(t)
                    corrected.append(c if c is not None else t)
                else:
                    corrected.append(t)
            tokens = corrected
        except Exception:
            # If SpellChecker not available, skip silently
            pass

    if do_pos_tag:
        return {'tokens': tokens, 'pos_tags': pos_tags}

    return tokens


def clean_and_preprocess(raw_text):
    """Backward-compatible simple pipeline: lowercase -> remove stopwords -> stem."""
    return preprocess_text(raw_text,
                           lowercase=True,
                           remove_punctuation=True,
                           remove_stopwords=True,
                           do_stemming=True)

if __name__ == '__main__':
    sample = "The boys are running and the leaves are falling."
    print('clean_and_preprocess:', clean_and_preprocess(sample))
    print('lemmatize + pos:', preprocess_text(sample, do_lemmatize=True, do_pos_tag=True))