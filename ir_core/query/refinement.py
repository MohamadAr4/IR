"""Query refinement (requirement #5): query-formulation assistance, query
suggestion, synonym expansion, spell correction and personalisation from the
user's search history.

These are the "additional features" that the evaluation (requirement #8) must
measure *before* and *after*, so each can be toggled independently. The refiner
is corpus-aware: corrections and expansion terms are only kept when they
actually occur in the index (df > 0), which keeps refinement from injecting
out-of-vocabulary noise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.corpus import words as nltk_words, wordnet

from ..index.inverted_index import InvertedIndex
from ..text.preprocessing import process, stem

for _res, _path in (("words", "corpora/words"), ("wordnet", "corpora/wordnet")):
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_res, quiet=True)
        except Exception:
            pass

try:
    _ENGLISH = set(w.lower() for w in nltk_words.words())
except Exception:
    _ENGLISH = set()

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in _LETTERS]
    inserts = [L + c + R for L, R in splits for c in _LETTERS]
    return set(deletes + transposes + replaces + inserts)


@dataclass
class RefinementResult:
    original: str
    corrected: str                       # raw string after spell-correction
    refined_raw: str                     # corrected text only (for dense/BERT)
    refined_tokens: list[str]            # corrected + synonym expansion, stemmed (for lexical)
    corrections: dict = field(default_factory=dict)   # {wrong: right}
    expansions: list = field(default_factory=list)     # synonym words added
    expansion_terms: list = field(default_factory=list)  # stemmed synonyms (lexical down-weighting)
    suggestions: list = field(default_factory=list)    # suggested full queries
    applied: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "original": self.original, "corrected": self.corrected,
            "refined_raw": self.refined_raw, "refined_tokens": self.refined_tokens,
            "corrections": self.corrections, "expansions": self.expansions,
            "expansion_terms": self.expansion_terms,
            "suggestions": self.suggestions, "applied": self.applied,
        }


class QueryRefiner:
    def __init__(self, index: InvertedIndex):
        self.index = index

    # -- spell correction -------------------------------------------------
    def _best_correction(self, word: str) -> Optional[str]:
        if word in _ENGLISH or len(word) <= 3 or not _ENGLISH:
            return None
        if self.index.df(stem(word)) > 0:        # already a known corpus term
            return None
        cands = {c for c in _edits1(word) if c in _ENGLISH}
        if not cands:
            cands = {c2 for c in _edits1(word) for c2 in _edits1(c) if c2 in _ENGLISH}
        if not cands:
            return None
        # rank by how strongly the candidate appears in *this* corpus
        best = max(cands, key=lambda c: self.index.df(stem(c)))
        return best if self.index.df(stem(best)) > 0 else None

    def correct(self, query: str) -> tuple[str, dict]:
        corrections: dict[str, str] = {}

        def repl(m):
            w = m.group(0)
            fix = self._best_correction(w.lower())
            if fix and fix != w.lower():
                corrections[w] = fix
                return fix
            return w

        return _WORD_RE.sub(repl, query), corrections

    # -- synonym expansion (WordNet) -------------------------------------
    def expand(self, query: str, max_per_word: int = 1) -> list[str]:
        added: list[str] = []
        seen = set(process(query))
        for w in _WORD_RE.findall(query.lower()):
            n = 0
            # Only the first synset = the most common sense. Iterating every
            # synset pulls in unrelated senses (e.g. "machine" -> "political
            # machine"), which is pure noise for retrieval.
            synsets = wordnet.synsets(w)
            if not synsets:
                continue
            for lemma in synsets[0].lemma_names():
                cand = lemma.replace("_", " ").lower()
                if " " in cand or cand == w:
                    continue
                st = stem(cand)
                if st in seen or self.index.df(st) == 0:
                    continue
                seen.add(st)
                added.append(cand)
                n += 1
                if n >= max_per_word:
                    break
        return added

    # -- suggestions ------------------------------------------------------
    def suggest(self, query: str, history: Optional[list[str]] = None,
                limit: int = 5) -> list[str]:
        suggestions: list[str] = []
        q_tokens = set(process(query))

        # 1) previous queries that share a term with the current one
        for past in (history or []):
            if past.strip().lower() == query.strip().lower():
                continue
            if q_tokens & set(process(past)):
                suggestions.append(past)

        # 2) complete the last word from the index vocabulary (popular terms)
        words = _WORD_RE.findall(query.lower())
        if words:
            for term in self.index.vocab_sample(stem(words[-1]), limit=limit):
                cand = (" ".join(words[:-1] + [term])).strip()
                if cand and cand != query.lower():
                    suggestions.append(cand)

        # dedupe, keep order
        out, seen = [], set()
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out[:limit]

    # -- history personalisation -----------------------------------------
    def history_terms(self, history: Optional[list[str]], query: str,
                      max_terms: int = 3) -> list[str]:
        """Pull a few salient terms from related past queries to bias retrieval
        toward the user's apparent intent."""
        if not history:
            return []
        q_tokens = set(process(query))
        extra: list[str] = []
        for past in reversed(history):
            toks = process(past)
            if q_tokens & set(toks):
                for t in toks:
                    if t not in q_tokens and t not in extra and self.index.df(t) > 0:
                        extra.append(t)
                        if len(extra) >= max_terms:
                            return extra
        return extra

    # -- one-shot refinement ---------------------------------------------
    def refine(self, query: str, *, do_spell: bool = True, do_expand: bool = True,
               use_history: bool = True, history: Optional[list[str]] = None,
               max_per_word: int = 1) -> RefinementResult:
        corrected, corrections = (self.correct(query) if do_spell else (query, {}))
        expansions = self.expand(corrected, max_per_word=max_per_word) if do_expand else []
        hist_terms = self.history_terms(history, corrected) if use_history else []

        # Synonyms help LEXICAL retrieval (literal term matching) but hurt DENSE
        # embeddings: the model already maps synonyms to nearby vectors, so
        # appending them only drags the query vector off-topic (query drift).
        # Hence the dense path (refined_raw) is spell-correction only, while the
        # synonym expansion is kept on the lexical tokens path.
        expansion_terms = [stem(w) for w in expansions]
        refined_raw = corrected
        refined_tokens = process(corrected) + expansion_terms + hist_terms

        return RefinementResult(
            original=query, corrected=corrected, refined_raw=refined_raw,
            refined_tokens=refined_tokens, corrections=corrections,
            expansions=expansions, expansion_terms=expansion_terms,
            suggestions=self.suggest(query, history=history),
            applied={"spell": do_spell, "expand": do_expand, "history": use_history,
                     "history_terms": hist_terms},
        )
