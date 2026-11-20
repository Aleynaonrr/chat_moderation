"""Area B - Minimum edit distance (Levenshtein) and a lexicon matcher (rule baseline).

The offensive lexicon is NOT hand-written: it is extracted automatically from the
TRAINING split only - words that appear much more often in offensive messages than
in clean ones (smoothed log-odds ratio). So no test data leaks into the rules.
"""
import json
import math
from collections import Counter
from pathlib import Path

from .normalize import normalize_tokens


def edit_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Levenshtein distance with dynamic programming (insert / delete / substitute = 1).
    Stops early and returns max_dist + 1 when the distance is surely larger than max_dist."""
    if a == b:
        return 0
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + cost)   # substitution
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def build_lexicon(texts, labels, top_k: int = 300, min_count: int = 5, min_len: int = 3) -> list[str]:
    """Pick the top_k words that are most indicative of the offensive class (log-odds, add-one smoothing)."""
    off, clean = Counter(), Counter()
    for t, y in zip(texts, labels):
        words = {w for w in normalize_tokens(t, fold=True) if w.isalpha() and len(w) >= min_len}
        (off if y == 1 else clean).update(words)
    n_off, n_clean = sum(off.values()), sum(clean.values())
    v = len(set(off) | set(clean))
    scores = {}
    for w in set(off) | set(clean):
        c_off, c_clean = off[w], clean[w]
        if c_off + c_clean < min_count or c_off <= c_clean:
            continue
        log_odds = math.log((c_off + 1) / (n_off + v)) - math.log((c_clean + 1) / (n_clean + v))
        scores[w] = log_odds * math.log(c_off + 1)  # reward both a high ratio and a high frequency
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


class LexiconMatcher:
    """Rule baseline: a message is OFFENSIVE if one of its words is close (edit distance)
    to a lexicon word, or starts with a lexicon word (Turkish suffixes: 'salaksın' -> 'salak')."""

    def __init__(self, lexicon: list[str]):
        self.lexicon = lexicon
        self.lex_set = set(lexicon)
        self._cache: dict[str, str | None] = {}

    @staticmethod
    def _allowed_dist(word: str) -> int:
        # short words must match exactly, longer words may contain 1-2 typos
        return 0 if len(word) <= 4 else (1 if len(word) <= 7 else 2)

    def match_word(self, w: str) -> str | None:
        if w in self._cache:
            return self._cache[w]
        found = w if w in self.lex_set else None
        if found is None:
            for lw in self.lexicon:
                if len(lw) >= 4 and w.startswith(lw):
                    found = lw
                    break
                d = self._allowed_dist(lw)
                if d and edit_distance(w, lw, max_dist=d) <= d:
                    found = lw
                    break
        self._cache[w] = found
        return found

    def matches(self, text: str) -> list[tuple[int, str, str]]:
        """List of (token index, token, matched lexicon word)."""
        out = []
        for i, w in enumerate(normalize_tokens(text, fold=True)):
            if w.isalpha() and len(w) >= 3:
                m = self.match_word(w)
                if m:
                    out.append((i, w, m))
        return out

    def predict_proba(self, texts) -> list[float]:
        """Pseudo-probability of OFF: 0 matches -> 0.0, 1 match -> 0.75, 2+ -> 1.0."""
        return [0.0 if not (n := len(self.matches(t))) else min(1.0, 0.5 + 0.25 * n) for t in texts]

    def save(self, path):
        Path(path).write_text(json.dumps(self.lexicon, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    print("edit_distance('salak', 'slak')   =", edit_distance("salak", "slak"))
    print("edit_distance('aptal', 'abtal')  =", edit_distance("aptal", "abtal"))
    print("edit_distance('kitap', 'kalem')  =", edit_distance("kitap", "kalem"))
    m = LexiconMatcher(["salak", "aptal"])
    for msg in ["sen tam bir s4l4k sın", "abtal misin", "bugün hava güzel"]:
        print(f"{msg!r:28} -> matches: {m.matches(msg)}")