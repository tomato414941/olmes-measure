"""Paloma's text (AI2, arXiv 2312.10523), read as Paloma reads it.

Paloma holds the validation and test text of 16 sources in 546 domains, published for evaluation. A domain's score is
the bits per byte of its documents, each read by itself after a BOS token and, when it is longer than the context,
in disjoint pieces (`pieces`, `rows`, `bits_per_byte`; `document_rows` keeps each document's rows apart). `fetch`
keeps the text of some sources and domains, a definition's members, as an artifact, and `part` takes a fixed part of
one that leaves out no member and no domain, for a reader that cannot read all of it at every measurement. A member is
a source, or SOURCE/DOMAIN for one domain of a source that mixes kinds (Dolma, RedPajama).

`part`: a member gives `member_bytes`, spread evenly over its domains, and a domain no less than `domain_bytes`. A
domain gives its documents in their order, each from its beginning for the domain's bytes spread evenly over its
documents and no less than `document_bytes` (about what a model reads at once, so that long documents are read far
from their beginnings too), until the domain's bytes are full. The part is chosen by the text's order alone.

`paragraphs` gives the paragraphs by which Paloma's guideline finds its text in other text (appendix C.1.1), which
overlap.py uses.

Paloma is gated: for `fetch`, HF_TOKEN must hold a token of a Hugging Face account that has accepted AI2's license.
"""
import collections
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import olmes_measure

REPOSITORY = "allenai/paloma"
REVISION = "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"
# A paragraph counts from this many tokens on, the length that Paloma takes from GPT-3's and Gopher's checks: shorter
# ones are in other text by chance. Paloma counts Unicode's segments; these are words and single marks.
PARAGRAPH_TOKENS = 13
TOKEN = re.compile(r"\w+|[^\w\s]")
ALIAS = "paloma:"  # what a reader may call a domain's text beside OLMES's tasks: paloma:SOURCE/DOMAIN


def domain(source, path):
    """The domain that a file of a source holds: NAME of `val_NAME.jsonl.gz`, or the source itself when it is one
    domain, whose files are `val.jsonl.gz` or the numbered parts `val-00000000.jsonl.gz`."""
    return Path(path).name.removesuffix(".jsonl.gz").partition("_")[2] or source


def listed(source, split):
    """The split's files of a source at REVISION, in the order of their names."""
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(REPOSITORY, repo_type="dataset", revision=REVISION, path_in_repo=f"{source}/{split}",
                                  token=os.environ["HF_TOKEN"])
    return sorted(item.path for item in tree if item.path.endswith(".jsonl.gz"))


def download(path, directory):
    """One file of Paloma at REVISION, fetched into the directory."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPOSITORY, path, repo_type="dataset", revision=REVISION, local_dir=directory,
                           token=os.environ["HF_TOKEN"])


def members(categories):
    """The members of some categories ({category: [member]}) as (source, domain), the domain empty for a whole
    source."""
    return [(source, name) for names in categories.values()
            for source, _, name in (member.partition("/") for member in names)]


def fetch(out, split, categories, download=download, listed=listed):
    """The artifact of a split's text (val or test): the files of the categories' members as Paloma gives them, under
    texts/SOURCE."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    wanted = collections.defaultdict(set)
    for source, name in members(categories):
        wanted[source].add(name)
    with tempfile.TemporaryDirectory() as staging:
        for source, names in sorted(wanted.items()):
            (work / "texts" / source).mkdir(parents=True)
            kept = [path for path in listed(source, split) if "" in names or domain(source, path) in names]
            missing = names - {""} - {domain(source, path) for path in kept}
            if missing or not kept:
                raise ValueError(f"{source} gives no {split} file of {sorted(missing) or 'its own'}")
            for path in kept:
                shutil.copyfile(download(path, staging), work / "texts" / source / Path(path).name)
    return olmes_measure.seal(work, f"paloma-{split}", repository=REPOSITORY, revision=REVISION,
                              categories={category: list(names) for category, names in categories.items()})


def files(artifact):
    """An artifact's files by what they hold: {(source, domain): its files in the order of their names}."""
    found = collections.defaultdict(list)
    for path in sorted((Path(artifact) / "texts").glob("*/*.jsonl.gz")):
        found[path.parent.name, domain(path.parent.name, path)].append(path)
    return dict(sorted(found.items()))


def documents(paths):
    """The texts of the documents of a domain's files, in their order."""
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)["text"]


def beginning(text, limit):
    """The beginning of a text within `limit` bytes of UTF-8. A text that is cut ends after the last space within
    them, so that no word is cut."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    cut = data[:limit].decode("utf-8", errors="ignore")
    return cut[:max(cut.rfind(" "), cut.rfind("\n")) + 1] or cut


def part(artifact, out, categories, member_bytes, domain_bytes, document_bytes):
    """The fixed part of an artifact that `fetch` made of these categories, as an artifact of its own with one file
    for each domain: of each domain the beginnings of its first documents, as the module's description gives them.
    Documents without text are passed over, and so is a domain without any."""
    artifact, out = Path(artifact), Path(out)
    manifest = olmes_measure.check(artifact)
    held = files(artifact)
    owner, domains = {}, collections.Counter()  # each domain's member, and how many domains a member holds
    for member in members(categories):
        for source, name in held:
            if source == member[0] and member[1] in ("", name):
                owner[source, name] = member
                domains[member] += 1
    if set(owner) != set(held) or set(domains) != set(members(categories)):
        raise ValueError(f"{artifact} does not hold the text of the categories' members, each domain once")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    for (source, name), paths in held.items():
        texts = [text for text in documents(paths) if text.strip()]
        if not texts:
            continue
        budget = max(member_bytes // domains[owner[source, name]], domain_bytes)
        share = max(budget // len(texts), document_bytes)
        kept, taken = [], 0
        for text in texts:
            if taken >= budget:
                break
            kept.append(beginning(text, min(share, budget - taken)))
            taken += len(kept[-1].encode("utf-8"))
        (work / "texts" / source).mkdir(parents=True, exist_ok=True)
        data = io.BytesIO()
        with gzip.GzipFile(fileobj=data, mode="wb", mtime=0) as file:  # no time in it: the same text, the same file
            for text in kept:
                file.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
        (work / "texts" / source / paths[0].name).write_bytes(data.getvalue())
    return olmes_measure.seal(work, f"{manifest['kind']}-sample", of=artifact.name, member_bytes=member_bytes,
                              domain_bytes=domain_bytes, document_bytes=document_bytes,
                              categories={category: list(names) for category, names in categories.items()})


def pieces(tokens, context):
    """The rows in which a model of `context` tokens reads a document's tokens, its BOS token first: the whole of
    it, or disjoint pieces, each after the last token of the piece before. A row is its tokens and how many of its
    last ones are scored, all but the first, as olmes-measure gives a question's row."""
    rows = [tokens[start:start + context + 1] for start in range(0, len(tokens) - 1, context)]
    return [(row, len(row) - 1) for row in rows]


def document_rows(artifact, encode, bos, context):
    """What a model of `context` tokens reads of an artifact, a domain at a time and a document at a time: (the
    domain's alias, [(a document's size in bytes, its rows as `pieces` gives them)]). `encode` gives texts' tokens
    without special tokens."""
    for (source, name), paths in files(artifact).items():
        texts, found = list(documents(paths)), []
        for at in range(0, len(texts), 2048):  # a few thousand at a time, as olmes-measure encodes its rows
            for text, tokens in zip(texts[at:at + 2048], encode(texts[at:at + 2048])):
                found.append((len(text.encode("utf-8")), pieces([bos, *tokens], context)))
        yield f"{ALIAS}{source}/{name}", found


def rows(artifact, encode, bos, context):
    """What a model of `context` tokens reads of an artifact, a domain at a time: (the domain's alias, its documents'
    rows as `pieces` gives them). `encode` gives texts' tokens without special tokens."""
    for alias, found in document_rows(artifact, encode, bos, context):
        yield alias, [row for _, made in found for row in made]


def bits_per_byte(nats, size):
    """The bits per byte of text of `size` bytes whose tokens' log-likelihoods add up to `nats`."""
    return -nats / (math.log(2) * size)


def paragraphs(text):
    """The paragraphs by which a document of Paloma's is found in other text: its lines of PARAGRAPH_TOKENS tokens or
    more, without the space around them. A line of marks alone, which formatting repeats, does not count."""
    for line in text.split("\n"):
        line = line.strip()
        if len(line) >= PARAGRAPH_TOKENS and re.search(r"\w", line) and len(TOKEN.findall(line)) >= PARAGRAPH_TOKENS:
            yield line


def digest(paragraph):
    return hashlib.blake2b(paragraph.encode("utf-8"), digest_size=16).digest()
