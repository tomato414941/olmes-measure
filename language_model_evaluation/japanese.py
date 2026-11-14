"""Japanese tasks asked as OLMES asks English ones for bits per byte, kept as a requests artifact.

OLMES measures a small model on a task by the bits per byte of each question's correct answer after the question
(its rc:bpb tasks): a score that moves with what the model knows rather than with how it answers. No public tool asks
Japanese tasks this way, so `make` does what OLMES does for the Japanese tasks that the language-model projects
measure, from each task's own data at a fixed revision:

- JCommonsenseQA (JGLUE): a question and its answer, after five questions of the training split with theirs;
- KUCI: a clause and the clause that follows it, after five pairs of the training split;
- xWinograd (Japanese): the rest of the sentence after the referent, as OLMES reads WinoGrande;
- JSQuAD (JGLUE): a passage, a question and the answer in it, after two of the training split;
- JMMLU: a question and its answer, each subject a task, as OLMES's MMLU, without examples: it has no other split;
- JEMHopQA, NIILC and AIO (JAQKET): a question and its short answer, after five of another split, as llm-jp-eval
  splits them (NIILC's first answer where it gives several, and none of its questions whose answer is "-");
- JamC-QA: a question about Japan and its answer of four, after five of its development split.

The tasks are those of llm-jp-eval's categories that a correct answer of a few words can score: commonsense
reasoning, reading comprehension, examinations and question answering.

The artifact has the form of olmes_build.py's, every choice of a question a request, so that olmes_measure's select,
rows and score read it unchanged. Its suite `ja:bpb` holds the tasks, JMMLU as the group of its subjects. A
request's `doc` holds its question as the task writes it, by which overlap.py finds the question in other text, and
the artifact says that its language is Japanese. Reading JamC-QA's files needs pyarrow.

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
import xml.etree.ElementTree
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
    "jemhopqa": {"url": GITHUB, "repository": "aiishii/JEMHopQA",
                 "revision": "fed5024f4f39f02727e699cb7ccb67935a779a5a", "license": "CC-BY-SA-4.0", "files": {
                     "corpus/train.json": "c64392b4087299bcfaa02fe2584128b5dc50fe5ea83e1bd600cadfc984669e65",
                     "corpus/dev.json": "2f3b8a518ca3051fc002f343ba3c26f6f3be902464061b43c30c5a1d37a9269a"}},
    "niilc": {"url": GITHUB, "repository": "mynlp/niilc-qa", "revision": "b6d47968b2e301a5e953fb1922a08ed56099646b",
              "license": "CC-BY-SA-4.0", "files": {
                  "data/NIILC-ECQA2015_dev.xml": "983dcaeb9ef38ca45d1114e8b420fdd4b358e827c235b1bfb4fc5e50ca70caf3",
                  "data/NIILC-ECQA2015_test.xml": "4a787463961db87e56e496c49cb40ca59f26d220e3ccc48588ad65b0dd4ce0d2"}},
    # JAQKET's files of the first AI King contest, which are not kept under revisions: their SHA-256 fix them.
    "aio": {"url": "https://jaqket.s3.ap-northeast-1.amazonaws.com/{path}", "repository": "jaqket", "revision": None,
            "license": "CC-BY-SA-4.0", "files": {
                "data/aio_02/aio_01_dev.jsonl": "ae62245b9aa20e641acdaac5811ef5037b3634a1385907f0c997619d9eda01b8",
                "data/aio_02/aio_01_test.jsonl": "ad4e1ba3125732f62bce2515abfff5d61cee9015f463ade9018b8023aadc0e18"}},
    "jamc_qa": {"url": HUGGING_FACE, "repository": "sbintuitions/JamC-QA",
                "revision": "ec2f8a34b1cfb9dcd8c50583eb688ef2ee3fb671", "license": "CC-BY-SA-4.0", "files": {
                    "v1.0/dev-00000-of-00001.parquet":
                        "b27ba65ba655716a6127be9a4f7e1051684318dec746fccba6cc276cda9b02e8",
                    "v1.0/test-00000-of-00001.parquet":
                        "c99a510c207b110c2beacb053cbd1d72cd4c593bfeced71478ff04a942f55bb6"}},
    "jmmlu": {"url": HUGGING_FACE, "repository": "nlp-waseda/JMMLU",
              "revision": "3637b25e444ccfdcde4d23a783cbe8e674faa01b", "license": "CC-BY-NC-ND-4.0", "files": {
                  "JMMLU.zip": "3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d"}},
}
# Where the sources' files are kept unless another place is named: in the user's cache, as Hugging Face keeps its own.
RAW = Path.home() / ".cache" / "language-model-evaluation" / "japanese"
SUITE = "ja:bpb"
JMMLU = "jmmlu:rc:bpb::ja"
QUESTION, ANSWER = "質問：", "回答："
SEED = 0  # which questions of a training split are a task's examples


def fetch(source, path, raw, sources=SOURCES):
    """A file of a source, fetched into `raw` once and checked against its SHA-256."""
    spec = sources[source]
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


def parquet_rows(path):
    """The rows of a parquet file, each a dict."""
    import pyarrow.parquet

    return pyarrow.parquet.read_table(path).to_pylist()


def shots(examples, count, task):
    """`count` examples of a training split, the same each time: the text before every question of the task."""
    chosen = random.Random(f"{SEED}:{task}").sample(examples, count) if count else []
    return "".join(f"{context}{answer}\n\n" for context, answer in chosen)


def chosen(context, choices, label, question):
    """A question asked of each of its choices: (contexts, continuations, label, the question as the task writes it)."""
    return [context] * len(choices), list(choices), label, question


def jcommonsenseqa(raw):
    def asked(row):
        return (f"{QUESTION}{row['question']}\n{ANSWER}", [row[f"choice{k}"] for k in range(5)], row["label"],
                row["question"])

    folder = "datasets/jcommonsenseqa-v1.2"
    train = [asked(row) for row in json_lines(fetch("jglue", f"{folder}/train-v1.2.json", raw))]
    before = shots([(context, choices[label]) for context, choices, label, _ in train], 5, "jcommonsenseqa")
    return [chosen(before + context, choices, label, question) for context, choices, label, question
            in map(asked, json_lines(fetch("jglue", f"{folder}/valid-v1.2.json", raw)))]


def kuci(raw):
    def asked(row):
        # KUCI writes its clauses with a space between words, which Japanese text does not.
        joined = lambda text: text.replace(" ", "")
        choices, label = [joined(row[f"choice_{k}"]) for k in "abcd"], "abcd".index(row["label"])
        # The clause with its right continuation, as the text that KUCI came from wrote it.
        return joined(row["context"]), choices, label, joined(row["context"]) + choices[label]

    train = [asked(row) for row in json_lines(fetch("kuci", "dataset/train.jsonl", raw))]
    before = shots([(context, choices[label]) for context, choices, label, _ in train], 5, "kuci")
    return [chosen(before + context, choices, label, question) for context, choices, label, question
            in map(asked, json_lines(fetch("kuci", "dataset/test.jsonl", raw)))]


def xwinograd(raw):
    def asked(row):
        # The two sentences differ in the referent alone: what follows it is the continuation, as OLMES and
        # lm-evaluation-harness read WinoGrande, scored after each sentence's own beginning.
        first, second = row["sentence1"].strip(), row["sentence2"].strip()
        same = 0
        while same < min(len(first), len(second)) and first[-1 - same] == second[-1 - same]:
            same += 1
        rest, label = first[len(first) - same:], int(row["answer"]) - 1
        return [first[:len(first) - same], second[:len(second) - same]], [rest, rest], label, (first, second)[label]

    return [asked(row) for row in json_lines(fetch("xwinograd", "test/xwinograd-ja.json", raw))]


def jsquad(raw):
    def asked(row):
        passage, question = row
        return (f"文脈：{passage}\n{QUESTION}{question['question']}\n{ANSWER}", question["answers"][0]["text"],
                question["question"])

    def questions(path):
        with open(fetch("jglue", path, raw), encoding="utf-8") as source:
            articles = json.load(source)["data"]
        # A passage begins with its article's title and [SEP], which the text does not hold.
        return [(paragraph["context"].split("[SEP]")[-1].strip(), question) for article in articles
                for paragraph in article["paragraphs"] for question in paragraph["qas"]]

    folder = "datasets/jsquad-v1.2"
    before = shots([asked(row)[:2] for row in questions(f"{folder}/train-v1.2.json")], 2, "jsquad")
    return [chosen(before + context, [answer], 0, question)
            for context, answer, question in map(asked, questions(f"{folder}/valid-v1.2.json"))]


def jmmlu(raw):
    """Each subject's questions, by subject."""
    subjects = {}
    with zipfile.ZipFile(fetch("jmmlu", "JMMLU.zip", raw)) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("JMMLU/test/") or not name.endswith(".csv"):
                continue
            rows = csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig"))
            subjects[Path(name).stem] = [chosen(f"{QUESTION}{row['question'].strip()}\n{ANSWER}",
                                                [row[k] for k in "ABCD"], "ABCD".index(row["answer"].strip()),
                                                row["question"].strip())
                                         for row in rows]
    return subjects


def answered(train, test, task):
    """Questions with a short answer, each after five of another split: (question, answer) pairs of each split."""
    asked = lambda question: f"{QUESTION}{question}\n{ANSWER}"
    before = shots([(asked(question), answer) for question, answer in train], 5, task)
    return [chosen(before + asked(question), [answer], 0, question) for question, answer in test]


def jemhopqa(raw):
    pairs = lambda path: [(row["question"], row["answer"])
                          for row in json.loads(fetch("jemhopqa", path, raw).read_text(encoding="utf-8"))]
    # As llm-jp-eval splits it: the development questions are the ones asked.
    return answered(pairs("corpus/train.json"), pairs("corpus/dev.json"), "jemhopqa")


def niilc(raw):
    def pairs(path):
        found = []
        for question in xml.etree.ElementTree.parse(fetch("niilc", path, raw)).getroot().iter("question"):
            answers = [answer.text for answer in question.find("answers").iter("answer") if answer.text]
            if answers and answers != ["-"]:  # "-": the question has no answer, and llm-jp-eval leaves it out
                found.append((question.find("text").text, answers[0]))
        return found

    return answered(pairs("data/NIILC-ECQA2015_dev.xml"), pairs("data/NIILC-ECQA2015_test.xml"), "niilc")


def aio(raw):
    pairs = lambda path: [(row["question"], row["answers"][0]) for row in json_lines(fetch("aio", path, raw))]
    return answered(pairs("data/aio_02/aio_01_dev.jsonl"), pairs("data/aio_02/aio_01_test.jsonl"), "aio")


def jamc_qa(raw):
    def asked(row):
        choices = [row[f"choice{k}"] for k in range(4)]
        return f"{QUESTION}{row['question']}\n{ANSWER}", choices, int(row["answer_index"]), row["question"]

    dev = [asked(row) for row in parquet_rows(fetch("jamc_qa", "v1.0/dev-00000-of-00001.parquet", raw))]
    before = shots([(context, choices[label]) for context, choices, label, _ in dev], 5, "jamc_qa")
    return [chosen(before + context, choices, label, question) for context, choices, label, question
            in map(asked, parquet_rows(fetch("jamc_qa", "v1.0/test-00000-of-00001.parquet", raw)))]


def requests(asked):
    """A task's requests as OLMES writes them: each question's choices in order, all naming the question."""
    return [{"doc_id": doc_id, "native_id": doc_id, "idx": idx, "label": label, "doc": {"question": question},
             "request": {"context": context, "continuation": continuation}}
            for doc_id, (contexts, continuations, label, question) in enumerate(asked)
            for idx, (context, continuation) in enumerate(zip(contexts, continuations))]


def make(out, raw):
    """The requests artifact of the Japanese tasks, in the directory `out`, from their files kept in `raw`."""
    tasks = {"jcommonsenseqa:rc:bpb::ja": (jcommonsenseqa(raw), {"split": "validation", "num_shots": 5}),
             "kuci:rc:bpb::ja": (kuci(raw), {"split": "test", "num_shots": 5}),
             "xwinograd_ja:rc:bpb::ja": (xwinograd(raw), {"split": "test", "num_shots": 0}),
             "jsquad:rc:bpb::ja": (jsquad(raw), {"split": "validation", "num_shots": 2}),
             "jemhopqa:rc:bpb::ja": (jemhopqa(raw), {"split": "dev", "num_shots": 5}),
             "niilc:rc:bpb::ja": (niilc(raw), {"split": "test", "num_shots": 5}),
             "aio:rc:bpb::ja": (aio(raw), {"split": "test", "num_shots": 5}),
             "jamc_qa:rc:bpb::ja": (jamc_qa(raw), {"split": "test", "num_shots": 5})}
    subjects = {f"jmmlu_{subject}:rc:bpb::ja": asked for subject, asked in jmmlu(raw).items()}
    tasks |= {alias: (asked, {"split": "test", "num_shots": 0}) for alias, asked in subjects.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "requests").mkdir()
    configs = {}
    for alias, (asked, settings) in sorted(tasks.items()):
        data = io.BytesIO()
        with gzip.GzipFile(fileobj=data, mode="wb", mtime=0) as file:  # no time in it: the same tasks, the same file
            file.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests(asked)).encode("utf-8"))
        (work / "requests" / f"{alias}.jsonl.gz").write_bytes(data.getvalue())
        configs[alias] = {"task_name": alias.partition(":")[0], **settings, "seed": SEED,
                          "primary_metric": "bits_per_byte_corr", "compute_gold_bpb": True}
    (work / "tasks.json").write_text(json.dumps(configs, indent=1))
    suites = {SUITE: sorted(set(tasks) - set(subjects)) + [JMMLU], JMMLU: sorted(subjects)}
    (work / "suites.json").write_text(json.dumps(suites, indent=1))
    sources = {name: {"repository": spec["repository"], "revision": spec["revision"]}
               for name, spec in SOURCES.items()}
    return olmes_measure.seal(work, "requests", olmes=None, made_by="language_model_evaluation.japanese",
                              language="ja", sources=sources, suites=sorted(suites))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("make")
    making.add_argument("out")
    making.add_argument("raw", nargs="?", default=RAW)
    args = parser.parse_args()
    print(make(args.out, args.raw))


if __name__ == "__main__":
    main()
