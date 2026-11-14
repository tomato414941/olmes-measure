"""Japanese tasks asked as OLMES asks English ones for bits per byte, kept as a requests artifact.

OLMES measures a small model on a task by the bits per byte of each question's correct answer after the question
(its rc:bpb tasks): a score that moves with what the model knows rather than with how it answers. No public tool asks
Japanese tasks this way, so `make` does what OLMES does for the Japanese tasks that the language-model projects
measure, from each task's own data at a fixed revision:

- JCommonsenseQA (JGLUE): a question and its answer, after five questions of the training split with theirs;
- KUCI: a clause and the clause that follows it, after five pairs of the training split;
- xWinograd (Japanese): the rest of the sentence after the referent, as OLMES reads WinoGrande;
- JSQuAD (JGLUE): a passage, a question and the answer in it, after two of the training split;
- JMMLU: a question and its answer, each subject a task, as OLMES's MMLU, without examples: it has no other split.

The artifact has the form of olmes_build.py's, every choice of a question a request, so that olmes_measure's select,
rows and score read it unchanged. Its suite `ja:bpb` holds the five tasks, JMMLU as the group of its subjects.

    python -m language_model_evaluation.japanese make OUT [RAW]   # RAW: where the tasks' files are kept
"""
import argparse
import csv
import gzip
import io
import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import olmes_measure

GITHUB = "https://raw.githubusercontent.com/{repository}/{revision}/{path}"
HUGGING_FACE = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"
# Each source's files at a fixed revision, with their SHA-256.
SOURCES = {
    "jglue": {"url": GITHUB, "repository": "yahoojapan/JGLUE", "revision": "c96c6af54200a404f31a3a4cd0b289786aa78809",
              "license": "CC-BY-SA-4.0", "files": {
                  "datasets/jcommonsenseqa-v1.2/train-v1.2.json":
                      "9b55fae5ecb3aedd6f8ce5bc09196c3b629864668ec6c18eee4d65c0aa48229e",
                  "datasets/jcommonsenseqa-v1.2/valid-v1.2.json":
                      "0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4",
                  "datasets/jsquad-v1.2/train-v1.2.json":
                      "24a0233a5d2cc80571e499de487fe2188981126312f159fc90136db82f3584b5",
                  "datasets/jsquad-v1.2/valid-v1.2.json":
                      "fb0a57a35281ff03bbe7f0e878cff5b577985fbe6fcb242e7e8ddb63d1c0ab11"}},
    "kuci": {"url": GITHUB, "repository": "ku-nlp/KUCI", "revision": "79afa1d63d1f262e6709e9be714d9ad5dae8d6f6",
             "license": "CC-BY-SA-4.0", "files": {
                 "dataset/train.jsonl": "77cd991d35cdc5cba23d0e2208cdc66e3173577449dffdf1503bf49e0624b076",
                 "dataset/test.jsonl": "d4741c5a786018e7094ed83be12bd71b812650ce347394d05d3725f9e8482a19"}},
    "xwinograd": {"url": HUGGING_FACE, "repository": "polm-stability/xwinograd-ja",
                  "revision": "2b841a0b715e00b389c38976daf25374368dee30", "license": "CC-BY-4.0", "files": {
                      "test/xwinograd-ja.json": "bc272a8bf08d52de84ee12bf1f575b440451e4ed450a8c7f6be8cf8e5cd9facd"}},
    "jmmlu": {"url": HUGGING_FACE, "repository": "nlp-waseda/JMMLU",
              "revision": "3637b25e444ccfdcde4d23a783cbe8e674faa01b", "license": "CC-BY-NC-ND-4.0", "files": {
                  "JMMLU.zip": "3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d"}},
}
SUITE = "ja:bpb"
JMMLU = "jmmlu:rc:bpb::ja"
QUESTION, ANSWER = "質問：", "回答："
SEED = 0  # which questions of a training split are a task's examples


def fetch(source, path, raw):
    """A file of a source, fetched into `raw` once and checked against its SHA-256."""
    spec = SOURCES[source]
    local = Path(raw) / source / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = spec["url"].format(repository=spec["repository"], revision=spec["revision"], path=path)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        partial = local.with_suffix(local.suffix + ".partial")
        partial.write_bytes(data)
        partial.rename(local)
    if olmes_measure.file_sha256(local) != spec["files"][path]:
        raise ValueError(f"{local} is not the file of {spec['repository']} at {spec['revision']}")
    return local


def json_lines(path):
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def shots(examples, count, task):
    """`count` examples of a training split, the same each time: the text before every question of the task."""
    chosen = random.Random(f"{SEED}:{task}").sample(examples, count) if count else []
    return "".join(f"{context}{answer}\n\n" for context, answer in chosen)


def chosen(context, choices, label):
    """A question asked of each of its choices: (contexts, continuations, label)."""
    return [context] * len(choices), list(choices), label


def jcommonsenseqa(raw):
    def asked(row):
        return f"{QUESTION}{row['question']}\n{ANSWER}", [row[f"choice{k}"] for k in range(5)], row["label"]

    folder = "datasets/jcommonsenseqa-v1.2"
    train = [asked(row) for row in json_lines(fetch("jglue", f"{folder}/train-v1.2.json", raw))]
    before = shots([(context, choices[label]) for context, choices, label in train], 5, "jcommonsenseqa")
    return [chosen(before + context, choices, label)
            for context, choices, label in map(asked, json_lines(fetch("jglue", f"{folder}/valid-v1.2.json", raw)))]


def kuci(raw):
    def asked(row):
        # KUCI writes its clauses with a space between words, which Japanese text does not.
        joined = lambda text: text.replace(" ", "")
        return joined(row["context"]), [joined(row[f"choice_{k}"]) for k in "abcd"], "abcd".index(row["label"])

    train = [asked(row) for row in json_lines(fetch("kuci", "dataset/train.jsonl", raw))]
    before = shots([(context, choices[label]) for context, choices, label in train], 5, "kuci")
    return [chosen(before + context, choices, label)
            for context, choices, label in map(asked, json_lines(fetch("kuci", "dataset/test.jsonl", raw)))]


def xwinograd(raw):
    def asked(row):
        # The two sentences differ in the referent alone: what follows it is the continuation, as OLMES and
        # lm-evaluation-harness read WinoGrande, scored after each sentence's own beginning.
        first, second = row["sentence1"].strip(), row["sentence2"].strip()
        same = 0
        while same < min(len(first), len(second)) and first[-1 - same] == second[-1 - same]:
            same += 1
        rest = first[len(first) - same:]
        return [first[:len(first) - same], second[:len(second) - same]], [rest, rest], int(row["answer"]) - 1

    return [asked(row) for row in json_lines(fetch("xwinograd", "test/xwinograd-ja.json", raw))]


def jsquad(raw):
    def asked(row):
        passage, question = row
        return f"文脈：{passage}\n{QUESTION}{question['question']}\n{ANSWER}", question["answers"][0]["text"]

    def questions(path):
        with open(fetch("jglue", path, raw), encoding="utf-8") as source:
            articles = json.load(source)["data"]
        # A passage begins with its article's title and [SEP], which the text does not hold.
        return [(paragraph["context"].split("[SEP]")[-1].strip(), question) for article in articles
                for paragraph in article["paragraphs"] for question in paragraph["qas"]]

    folder = "datasets/jsquad-v1.2"
    before = shots([asked(row) for row in questions(f"{folder}/train-v1.2.json")], 2, "jsquad")
    return [chosen(before + context, [answer], 0)
            for context, answer in map(asked, questions(f"{folder}/valid-v1.2.json"))]


def jmmlu(raw):
    """Each subject's questions, by subject."""
    subjects = {}
    with zipfile.ZipFile(fetch("jmmlu", "JMMLU.zip", raw)) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("JMMLU/test/") or not name.endswith(".csv"):
                continue
            rows = csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig"))
            subjects[Path(name).stem] = [chosen(f"{QUESTION}{row['question'].strip()}\n{ANSWER}",
                                                [row[k] for k in "ABCD"], "ABCD".index(row["answer"].strip()))
                                         for row in rows]
    return subjects


def requests(asked):
    """A task's requests as OLMES writes them: each question's choices in order, all naming the question."""
    return [{"doc_id": doc_id, "native_id": doc_id, "idx": idx, "label": label,
             "request": {"context": context, "continuation": continuation}}
            for doc_id, (contexts, continuations, label) in enumerate(asked)
            for idx, (context, continuation) in enumerate(zip(contexts, continuations))]


def make(out, raw):
    """The requests artifact of the Japanese tasks, in the directory `out`, from their files kept in `raw`."""
    tasks = {"jcommonsenseqa:rc:bpb::ja": (jcommonsenseqa(raw), {"split": "validation", "num_shots": 5}),
             "kuci:rc:bpb::ja": (kuci(raw), {"split": "test", "num_shots": 5}),
             "xwinograd_ja:rc:bpb::ja": (xwinograd(raw), {"split": "test", "num_shots": 0}),
             "jsquad:rc:bpb::ja": (jsquad(raw), {"split": "validation", "num_shots": 2})}
    subjects = {f"jmmlu_{subject}:rc:bpb::ja": asked for subject, asked in jmmlu(raw).items()}
    tasks |= {alias: (asked, {"split": "test", "num_shots": 0}) for alias, asked in subjects.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "requests").mkdir()
    configs = {}
    for alias, (asked, settings) in sorted(tasks.items()):
        with gzip.open(work / "requests" / f"{alias}.jsonl.gz", "wt", encoding="utf-8") as lines:
            lines.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in requests(asked))
        configs[alias] = {"task_name": alias.partition(":")[0], **settings, "seed": SEED,
                          "primary_metric": "bits_per_byte_corr", "compute_gold_bpb": True}
    (work / "tasks.json").write_text(json.dumps(configs, indent=1))
    suites = {SUITE: sorted(set(tasks) - set(subjects)) + [JMMLU], JMMLU: sorted(subjects)}
    (work / "suites.json").write_text(json.dumps(suites, indent=1))
    sources = {name: {"repository": spec["repository"], "revision": spec["revision"]}
               for name, spec in SOURCES.items()}
    return olmes_measure.seal(work, "requests", olmes=None, made_by="language_model_evaluation.japanese",
                              sources=sources, suites=sorted(suites))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("make")
    making.add_argument("out")
    making.add_argument("raw", nargs="?", default=Path.home() / ".cache" / "language-model-evaluation" / "japanese")
    args = parser.parse_args()
    print(make(args.out, args.raw))


if __name__ == "__main__":
    main()
