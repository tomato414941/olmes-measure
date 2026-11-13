"""The documents that overlap a benchmark's text, which nothing that is measured by it may learn or look up.

A project that is judged by OLMES's questions and Paloma's text keeps the documents that hold them out of what a
model learns from, what an index holds and what a tokenizer is made from. A document overlaps when it holds

- a question of OLMES's suites, found by its middle characters: the WIDTH characters in the middle of the question's
  letters and digits (without case, marks and space), when the document's own letters and digits hold them (the rule
  by which language-model's v6 corpus leaves such documents out);
- a paragraph of Paloma's text: a line of the document is one of Paloma's lines of 13 tokens or more
  (paloma.paragraphs), as Paloma's guideline finds its text; or
- a stretch of Paloma's text written another way: its letters and digits hold LETTERS in a row of a Paloma
  document's, as they stand at every STRIDE-th of that document's. A stretch of LETTERS + STRIDE letters and digits
  that the two share is then found however it is spaced, cased, marked or broken into lines. Paloma's WikiText is
  spaced around its marks, so a copy of the same article from Wikipedia shares no line with it, and a book's lines
  may be broken elsewhere: the line test alone passes both.

`make` keeps what tells the documents as an artifact small enough to go to every machine: the keys of every question
of a requests artifact, and of artifacts of Paloma's text (paloma.fetch: the splits whole, not only what a project
reads of them) the paragraphs' digests and the stretches' hashes. `load(artifact).holds(text)` tells a document; how
a project reads its training text, and what it counts as one document, is the project's.
"""
import atexit
import gzip
import itertools
import json
import multiprocessing
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

import olmes_measure

from . import paloma

# The rule by which language-model's v6 corpus leaves out the documents that hold evaluation questions.
PREAMBLE = re.compile(r"^The following are questions \(with answers\) about [^\n]*\n\n")
WRAPPER = re.compile(r"^(?:Question:|質問：)\s*|\s*(?:Answer:|回答：)$")
WIDTH, MINIMUM = 64, 24
# Where OLMES's document of a question holds the question alone: most tasks' query, or the field that it is made from.
QUESTION_FIELDS = ("query", "question", "text", "sentence")
# The stretches of Paloma's letters and digits that are looked for: so many in a row, from every so many-th.
LETTERS, STRIDE = 64, 128
# A stretch's hash is the sum of its characters' codes, each times a power of MULTIPLIER, modulo 2^64; with the
# multiplier's inverse, the hashes of all the stretches of a text come from one running sum.
MULTIPLIER = 0x9E3779B97F4A7C15
INVERSE = pow(MULTIPLIER, -1, 1 << 64)


# ASCII's bytes that are no small letter or digit: of a text of ASCII alone in small letters, what is not kept
DROPPED = bytes(byte for byte in range(128) if not chr(byte).isdigit() and not chr(byte).islower())


def normalized(text):
    """A text's letters and digits, without case: Unicode's compatible forms folded, and no marks, space or
    underscores. A text of ASCII alone, as most are, is done by its bytes, which is many times as fast."""
    if text.isascii():
        return text.lower().encode().translate(None, DROPPED).decode()
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def _powers(base, count):
    values = np.full(count, base, dtype=np.uint64)
    values[0] = 1
    return np.cumprod(values)  # modulo 2^64, as unsigned integers multiply


POWERS, INVERSES = _powers(MULTIPLIER, 1 << 16), _powers(INVERSE, 1 << 16)


def stretches(letters, width=LETTERS):
    """The hash of every `width` characters in a row of a text's letters and digits (normalized), in their order:
    the sum of code[i + k] x MULTIPLIER^k over the stretch that begins at i."""
    global POWERS, INVERSES
    codes = np.frombuffer(letters.encode("utf-32-le"), dtype="<u4").astype(np.uint64)
    if len(codes) < width:
        return np.empty(0, dtype=np.uint64)
    if len(codes) > len(POWERS):
        POWERS, INVERSES = _powers(MULTIPLIER, 2 * len(codes)), _powers(INVERSE, 2 * len(codes))
    running = np.cumsum(codes * POWERS[:len(codes)])
    sums = running[width - 1:].copy()
    sums[1:] -= running[:-width]
    return sums * INVERSES[:len(sums)]


def question_key(question):
    """The middle characters by which a question is found, or None if it is too short to tell from other text."""
    text = normalized(WRAPPER.sub("", PREAMBLE.sub("", question)))
    if len(text) < MINIMUM:
        return None
    width = min(len(text), WIDTH)
    middle = (len(text) - width) // 2
    return text[middle:middle + width]


def olmes_questions(requests):
    """The text of each question in an olmes-measure requests artifact, as OLMES's document for it gives it."""
    if olmes_measure.check(requests)["kind"] != "requests":
        raise ValueError(f"{requests} is no requests artifact")
    questions = set()
    for path in sorted(Path(requests, "requests").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                document = json.loads(line)["doc"]
                field = next((field for field in QUESTION_FIELDS if isinstance(document.get(field), str)), None)
                if field is None:
                    raise ValueError(f"a document of {path.name} holds none of {', '.join(QUESTION_FIELDS)}")
                questions.add(document[field])
    return questions


def keys_of(questions):
    """The questions' keys, in order: what `Overlap` finds questions by."""
    return sorted({question_key(question) for question in questions} - {None})


def digests_of(paragraphs):
    """The paragraphs' digests, in order and end to end (16 bytes each): what `Overlap` finds paragraphs by."""
    return b"".join(sorted({paloma.digest(paragraph) for paragraph in paragraphs}))


def stretches_of(texts):
    """The hashes of the stretches that are looked for of these documents (LETTERS in a row from every STRIDE-th of
    each one's letters and digits), in order and end to end (8 bytes each): what `Overlap` finds text written
    another way by."""
    found = [stretches(normalized(text))[::STRIDE] for text in texts]
    return np.unique(np.concatenate(found or [np.empty(0, dtype=np.uint64)])).astype("<u8").tobytes()


class Overlap:
    """What tells a document that overlaps the evaluation, from the questions' keys (keys_of), the paragraphs'
    digests (digests_of) and the hashes of the text's stretches of letters (stretches_of)."""

    def __init__(self, keys, digests, hashes=b""):
        import ahocorasick

        self.automaton, self.pools = None, {}  # the processes of `among`, by how many they are
        for key in keys:
            self.automaton = self.automaton or ahocorasick.Automaton()
            self.automaton.add_word(key, key)
        if self.automaton is not None:
            self.automaton.make_automaton()
        # the digests' two halves: a line is looked up by its first half, and is a paragraph if the second is the same
        halves = np.frombuffer(digests, dtype=">u8").astype(np.uint64).reshape(-1, 2)
        self.first, self.second = halves[:, 0].copy(), halves[:, 1].copy()
        if (self.first[1:] <= self.first[:-1]).any():
            raise ValueError("the paragraphs' digests are not in order, or two begin alike")
        self.hashes = np.frombuffer(hashes, dtype="<u8").astype(np.uint64)
        if (self.hashes[1:] <= self.hashes[:-1]).any():
            raise ValueError("the stretches' hashes are not in order")

    def holds_question(self, text, letters=None):
        letters = normalized(text) if letters is None else letters
        return self.automaton is not None and next(self.automaton.iter(letters), None) is not None

    def holds_stretch(self, text, letters=None):
        """Whether the document's letters and digits hold one of the stretches of Paloma's."""
        found = stretches(normalized(text) if letters is None else letters)
        if not len(found) or not len(self.hashes):
            return False
        at = np.minimum(np.searchsorted(self.hashes, found), len(self.hashes) - 1)
        return bool((self.hashes[at] == found).any())

    def holds_paragraph(self, text):
        lines = (line.strip() for line in text.split("\n"))
        found = [paloma.digest(line) for line in lines if len(line) >= paloma.PARAGRAPH_TOKENS]
        if not found or not len(self.first):
            return False
        halves = np.frombuffer(b"".join(found), dtype=">u8").astype(np.uint64).reshape(-1, 2)
        at = np.minimum(np.searchsorted(self.first, halves[:, 0]), len(self.first) - 1)
        return bool(((self.first[at] == halves[:, 0]) & (self.second[at] == halves[:, 1])).any())

    def holds(self, text):
        """Whether a document overlaps the evaluation."""
        letters = normalized(text)
        return (self.holds_question(text, letters) or self.holds_paragraph(text)
                or self.holds_stretch(text, letters))

    def among(self, workers):
        """A function that tells of each of some documents whether it overlaps the evaluation, the documents divided
        among so many processes: for reading billions of tokens, as a corpus index does. The processes are forked
        the first time, with what this holds, and stay until the program ends."""
        global FORKED
        if workers not in self.pools:
            FORKED = self
            self.pools[workers] = multiprocessing.get_context("fork").Pool(workers)
            atexit.register(self.pools[workers].terminate)  # while the program can still end them in order
        pool = self.pools[workers]
        return lambda texts: pool.map(_holds, texts, chunksize=max(1, len(texts) // (4 * workers)))


FORKED = None  # what the processes of `among` tell by, which they take with them when they are forked


def _holds(text):
    return FORKED.holds(text)


def make(out, requests, texts):
    """The artifact of what tells the documents that overlap the evaluation: the keys of the questions of an
    olmes-measure requests artifact (questions.txt, a line each), and of artifacts of Paloma's text the digests of
    the paragraphs (paragraphs.bin, 16 bytes each) and the hashes of the stretches of letters (stretches.bin, 8 bytes
    each), all in order."""
    for artifact in texts:
        if olmes_measure.check(artifact)["revision"] != paloma.REVISION:
            raise ValueError(f"{artifact} was not prepared from Paloma {paloma.REVISION}")
    keys = keys_of(olmes_questions(requests))
    documents = lambda: (text for artifact in texts for paths in paloma.files(artifact).values()
                         for text in paloma.documents(paths))
    digests = digests_of(paragraph for text in documents() for paragraph in paloma.paragraphs(text))
    return write(out, keys, digests, stretches_of(documents()), questions=Path(requests).name,
                 text=sorted(Path(artifact).name for artifact in texts))


def write(out, keys, digests, hashes=b"", **made_from):
    """The artifact of these keys, digests and hashes, in the directory `out`."""
    Overlap(keys, digests, hashes)  # what `load` will read
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "questions.txt").write_text("".join(key + "\n" for key in keys), encoding="utf-8")
    (work / "paragraphs.bin").write_bytes(digests)
    (work / "stretches.bin").write_bytes(hashes)
    return olmes_measure.seal(work, "overlap", width=WIDTH, minimum=MINIMUM, paragraph_tokens=paloma.PARAGRAPH_TOKENS,
                              letters=LETTERS, stride=STRIDE, **made_from)


LOADED = {}  # each artifact read once in a process


def load(artifact):
    """What an artifact of `make` tells."""
    artifact = Path(artifact)
    if artifact not in LOADED:
        olmes_measure.check(artifact)
        LOADED[artifact] = Overlap((artifact / "questions.txt").read_text(encoding="utf-8").split(),
                                   (artifact / "paragraphs.bin").read_bytes(),
                                   (artifact / "stretches.bin").read_bytes())
    return LOADED[artifact]
