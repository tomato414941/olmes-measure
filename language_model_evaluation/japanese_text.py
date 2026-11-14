"""Japanese text held out of the corpora it comes from, kept as Paloma keeps its text.

Paloma (paloma.py) measures how well a model reads English text of several kinds, from the validation and test splits
of public corpora. `fetch` keeps the held-out Japanese text of two public corpora as an artifact of the form of
paloma.fetch's (texts/SOURCE/*.jsonl.gz, a document {"text"} a line), so that paloma.part, paloma.rows and the
overlap test read it unchanged:

- mc4_ja, the web: mC4's Japanese validation split, as AllenAI publishes it (ODC-BY). mC4 has no Japanese test split,
  so its first four files are the validation text and its last four the test text, kept aside;
- wiki40b_ja, an encyclopedia: Wiki-40B's Japanese validation and test splits, articles of Japanese Wikipedia, with
  their markers of title, section and paragraph written as the lines of a text.

Reading Wiki-40B's files needs pyarrow.
"""
import ast
import gzip
import io
import json
import tempfile
from pathlib import Path

import olmes_measure

from .japanese import HUGGING_FACE, fetch as fetch_file

MC4 = "multilingual/c4-ja-validation.tfrecord-{:05d}-of-00008.json.gz"
SOURCES = {
    "mc4_ja": {"url": HUGGING_FACE, "repository": "allenai/c4", "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
               "license": "ODC-BY", "files": {MC4.format(n): digest for n, digest in enumerate([
                   "9891d068f288c4211e8d4b4187b66ce1a711983052c6e3745292cec1be951d26",
                   "eb0467da37b65ee133b664c18495c5bc3e219e1ad23af3c97cbd14bd20e887b0",
                   "6a95cc89dda237c90275da34851524542df4dbfffde5627af65351820d1c8102",
                   "d2aec1be6ca1168f02e716b84fc49e394e87efdf269ba604a3a3c1754cbd271d",
                   "b1a71e4440a897bfda028af245778929c24b0f5afe767fd51ca2cdc06512cc5d",
                   "36bce53dd11fb5eb6a96f282f77641e9694acec70a9964a350a3ef6d8325c303",
                   "8f65581f6b5a59e903696183e060718bebe43b407529cb2ceaad1a8266a2486e",
                   "1be9e08e6b95278c2a39fbeacd2064270d0c9c1ba04264d6fe519e6428ca63ff"])}},
    "wiki40b_ja": {"url": HUGGING_FACE, "repository": "google/wiki40b",
                   "revision": "06bfe584d8193cf58f79c6354ecc2a2b94a7deea", "license": "Wikipedia's (CC-BY-SA)",
                   "files": {
                       "ja/validation-00000-of-00001.parquet":
                           "7941561557a41410b8e69d804ab7c0e5cb261c5d478dbef02569d2978928b874",
                       "ja/test-00000-of-00001.parquet":
                           "85d1d3945748e5918566d5186b004622d85868a33ac4dcfcccf6627a6b377d6e"}},
}
# Each split's files of each source.
SPLITS = {"val": {"mc4_ja": [MC4.format(n) for n in range(4)], "wiki40b_ja": ["ja/validation-00000-of-00001.parquet"]},
          "test": {"mc4_ja": [MC4.format(n) for n in range(4, 8)], "wiki40b_ja": ["ja/test-00000-of-00001.parquet"]}}
MARKERS = ("_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_")


def article(text):
    """A Wiki-40B article as a text: its title, then each section's title and paragraphs, a line each. Wiki-40B's
    files hold each article as the written form of its UTF-8 bytes, and mark a line break inside a paragraph
    _NEWLINE_."""
    if text.startswith(("b'", 'b"')):
        text = ast.literal_eval(text).decode("utf-8")
    lines = [line for line in text.split("\n") if line and line not in MARKERS]
    return "\n".join(line.replace("_NEWLINE_", "\n") for line in lines)


def documents(source, path):
    """The texts of a file of a source, in its order."""
    if source == "mc4_ja":
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)["text"]
    else:
        import pyarrow.parquet

        for text in pyarrow.parquet.read_table(path, columns=["text"]).column("text").to_pylist():
            yield article(text)


def fetch(out, split, raw):
    """The artifact of a split's Japanese text (val or test), from the sources' files kept in `raw`: each file's
    documents under texts/SOURCE, in the files' order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    for source, paths in SPLITS[split].items():
        (work / "texts" / source).mkdir(parents=True)
        for number, path in enumerate(paths):
            data = io.BytesIO()
            with gzip.GzipFile(fileobj=data, mode="wb", mtime=0) as file:  # no time in it: the same text, the same file
                for text in documents(source, fetch_file(source, path, raw, SOURCES)):
                    file.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
            (work / "texts" / source / f"{split}-{number:08d}.jsonl.gz").write_bytes(data.getvalue())
    sources = {source: {"repository": SOURCES[source]["repository"], "revision": SOURCES[source]["revision"],
                        "files": SPLITS[split][source]} for source in SPLITS[split]}
    return olmes_measure.seal(work, f"japanese-text-{split}", language="ja", sources=sources)


def made_here(manifest):
    """Whether an artifact's manifest is of a split that `fetch` makes from the sources at their revisions."""
    split = manifest["kind"].removeprefix("japanese-text-")
    return split in SPLITS and manifest.get("sources") == {
        source: {"repository": SOURCES[source]["repository"], "revision": SOURCES[source]["revision"],
                 "files": SPLITS[split][source]} for source in SPLITS[split]}
