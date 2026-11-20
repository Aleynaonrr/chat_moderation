"""Area B - Classical NLP: regex-based normalization of noisy chat text.

Handles:
  * URLs and user mentions  -> <url>, @user
  * leetspeak               -> s4l4k -> salak, @ -> a, $ -> s (only in tokens that contain letters)
  * censored spelling       -> s.a.l.a.k / a*p*t*a*l -> salak / aptal
  * repeated letters        -> çoooook -> çok
  * Turkish lower-casing    -> I -> ı, İ -> i
  * Turkish char folding    -> ç->c, ğ->g, ı->i, ö->o, ş->s, ü->u (used for lexicon matching)
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(\w)\1{2,}")          # 3+ identical letters -> 1 letter
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1+")   # !!!! -> !
SPACES_RE = re.compile(r"\s+")
# tokens that mix letters with digits/symbols (leetspeak candidates)
LEET_TOKEN_RE = re.compile(r"\b(?=\w*[a-zçğıöşü])[\w@$]*[0-9@$][\w@$]*\b", re.IGNORECASE)
# letters separated by . * _ - (e.g. s.a.l.a.k) -> join them
SPLIT_LETTERS_RE = re.compile(r"\b(?:\w[.*_\-]){2,}\w\b")
TOKEN_RE = re.compile(r"<url>|@user|[a-zçğıöşüâîû]+|\d+|[^\w\s]", re.IGNORECASE)

LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                          "7": "t", "8": "b", "@": "a", "$": "s"})
TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
TR_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
                         "â": "a", "î": "i", "û": "u"})


def tr_lower(text: str) -> str:
    """Turkish-aware lower-casing (Python's lower() turns 'I' into 'i', not 'ı')."""
    return text.translate(TR_LOWER).lower()


def fold_turkish(text: str) -> str:
    """Map Turkish characters to ASCII so 'çok' and 'cok' match."""
    return text.translate(TR_FOLD)


def _deleet(match: re.Match) -> str:
    tok = match.group(0)
    return tok if tok.isdigit() else tok.translate(LEET_MAP)


def normalize(text: str, lower: bool = True, fold: bool = False) -> str:
    """Normalize a noisy chat message.

    lower=False -> keep casing (for the cased BERTurk model)
    fold=True   -> fold Turkish characters to ASCII (for the lexicon matcher)
    """
    t = str(text)
    t = URL_RE.sub(" <url> ", t)
    t = USER_RE.sub(" @user ", t)
    t = SPLIT_LETTERS_RE.sub(lambda m: re.sub(r"[.*_\-]", "", m.group(0)), t)
    t = LEET_TOKEN_RE.sub(_deleet, t)
    t = REPEAT_RE.sub(r"\1", t)
    t = PUNCT_REPEAT_RE.sub(r"\1", t)
    if lower:
        t = tr_lower(t)
    if fold:
        t = fold_turkish(t)
    return SPACES_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    """Split normalized text into word / punctuation tokens."""
    return TOKEN_RE.findall(text)


def normalize_tokens(text: str, fold: bool = False) -> list[str]:
    return tokenize(normalize(text, lower=True, fold=fold))

def norm_text(t: str) -> str:
    """Preprocessor for the tf-idf vectorizers (module-level so the pipeline can be pickled)."""
    return normalize(t, lower=True, fold=False)
if __name__ == "__main__":
    examples = ["Çoooooook GÜZELLL!!!!", "s4l4k misin @ahmet http://x.com", "IŞIK İYİ", "a.p.t.a.l"]
    for s in examples:
        print(f"{s!r:40} -> {normalize(s)!r:30} | folded: {normalize(s, fold=True)!r}")


