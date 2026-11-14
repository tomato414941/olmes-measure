"""Measure a language model on AI2's OLMES suites, reading only the requests that the suites' scores use.

OLMES (github.com/allenai/olmes) defines each task: its questions, examples and prompts, how its score is computed and
how a suite averages its tasks. olmes_build.py runs OLMES's own runner where OLMES is installed (olmes.txt) and keeps
every request of some suites as a requests artifact. Everything here works from that artifact without OLMES:

- select: the questions of some suites, each with only the requests that its task's score reads (the correct answer
  alone for bits per byte, every choice for accuracy), or a fixed random share of each task's questions;
- reserve: a selection of the questions that another selection does not hold, to be kept aside from decisions;
- within: where a selection's questions stand in a larger one, so that a result on the larger scores the smaller;
- rows: a selection's requests in a tokenizer, as OLMES's Hugging Face wrapper gives them to a model;
- score: each task's and each suite's score, as OLMES computes them, from a model's logits on those rows;
- values: each question's value under its task's metric, of which the task's score is the mean.

An artifact is a directory with manifest.json: what it was made from and the SHA-256 of each of its files. It is named
by its kind and the start of the SHA-256 of its manifest, so that equal artifacts have equal names whoever made them.

    python olmes_measure.py select REQUESTS OUT SUITE... [--per-task N] [--seed S]
    python olmes_measure.py reserve REQUESTS SELECTION OUT --per-task N [--seed S]
    python olmes_measure.py rows SELECTION OUT TOKENIZER_JSON BOS
    python olmes_measure.py check ARTIFACT...
"""

import argparse
import gzip
import hashlib
import itertools
import json
import random
import shutil
import tempfile
from pathlib import Path

# The longest row that OLMES's wrapper reads, as its maximum length is set here; longer rows lose their start.
LENGTH = 4096
# OLMES's value of log2(e), with which it turns a log-likelihood into bits.
LOG_2_OF_E = 1.44269504089
# What OLMES adds to a question's doc_id for the requests that score its choices without the question.
UNCONDITIONED = 1_000_000
# The fields of a task's configuration that say how it is scored rather than what it asks.
SCORING_FIELDS = ("primary_metric", "compute_gold_bpb", "metadata")
# The requests that each primary metric reads of a question: the one at its label, its first, or every choice, and
# with "uncond" the same choices without the question.
READS = {"bits_per_byte_corr": "label", "greedy_acc": "first", "acc_raw": "choices", "acc_per_token": "choices",
         "acc_per_char": "choices", "acc_per_byte": "choices", "acc_uncond": "uncond"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def seal(directory, kind, **made_from):
    """Writes directory/manifest.json and moves the directory to its artifact's name beside it. An artifact of the
    same name already there is the same one, which is kept."""
    directory = Path(directory)
    files = {str(path.relative_to(directory)): file_sha256(path) for path in sorted(directory.rglob("*"))
             if path.is_file() and path.name != "manifest.json"}
    data = canonical({"kind": kind, **made_from, "files": files})
    (directory / "manifest.json").write_bytes(data)
    target = directory.parent / f"{kind}-{hashlib.sha256(data).hexdigest()[:16]}"
    if target.exists():
        check(target)
        shutil.rmtree(directory)
    else:
        directory.rename(target)
    return target


def check(artifact):
    """An artifact's manifest, after checking its name and every file against it."""
    artifact = Path(artifact)
    data = (artifact / "manifest.json").read_bytes()
    manifest = json.loads(data)
    if artifact.name != f"{manifest['kind']}-{hashlib.sha256(data).hexdigest()[:16]}":
        raise ValueError(f"{artifact} is not named by its manifest")
    present = {str(path.relative_to(artifact)) for path in artifact.rglob("*") if path.is_file()}
    if present - {"manifest.json"} != set(manifest["files"]):
        raise ValueError(f"{artifact} holds other files than its manifest lists")
    for name, digest in manifest["files"].items():
        if file_sha256(artifact / name) != digest:
            raise ValueError(f"{artifact / name} differs from its manifest")
    return manifest


def leaves(groups, name):
    return [name] if name not in groups else [task for member in groups[name] for task in leaves(groups, member)]


def reached(groups, names):
    """The groups that the named suites hold, at any depth."""
    kept = {}
    for name in names:
        if name in groups and name not in kept:
            kept[name] = groups[name]
            kept.update(reached(groups, groups[name]))
    return kept


def questions(requests):
    """Each question's requests as OLMES's MCAccuracy groups them: (label, its requests, the same without the
    question or None), in the order of the questions' doc_ids. The requests without the question come after all
    the others, in their order."""
    asked = [request for request in requests if request["doc_id"] < UNCONDITIONED]
    unasked = requests[len(asked):]
    if unasked and [(r["doc_id"], r["idx"]) for r in unasked] != [(r["doc_id"] + UNCONDITIONED, r["idx"]) for r in asked]:
        raise ValueError("the requests without their question do not follow the ones with it in the same order")
    partner = dict(zip(map(id, asked), unasked))
    grouped = []
    for _, group in itertools.groupby(sorted(asked, key=lambda r: r["doc_id"]), key=lambda r: r["doc_id"]):
        group = list(group)
        if any(r["native_id"] != group[0]["native_id"] or r["label"] != group[0]["label"] for r in group):
            raise ValueError("the requests of one question name different questions")
        if [r["idx"] for r in group] != list(range(len(group))):
            raise ValueError("a question's choices are not in the order of their indices")
        grouped.append((group[0]["label"], group, [partner[id(r)] for r in group] if unasked else None))
    return grouped


def pair(request):
    return [request["request"]["context"], request["request"]["continuation"]]


def select(requests, out, suites, per_task=None, seed=0):
    """A selection artifact in the directory `out` from a requests artifact: for each task of the suites, its
    questions (or a fixed random `per_task` of them, the same for tasks that ask the same) with the requests that its
    task's primary metric reads."""
    made = check(requests)
    requests = Path(requests)
    groups = json.loads((requests / "suites.json").read_text())
    configs = json.loads((requests / "tasks.json").read_text())
    if set(suites) - set(groups):
        raise ValueError(f"{requests} holds no suite {sorted(set(suites) - set(groups))}")
    kept = reached(groups, suites)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "tasks").mkdir()
    metrics = {}
    for alias in sorted({task for suite in suites for task in leaves(groups, suite)}):
        config = configs[alias]
        metric = metrics[alias] = config["primary_metric"]
        if metric not in READS:
            raise ValueError(f"{alias} is scored by {metric}, which score does not compute")
        with gzip.open(requests / "requests" / f"{alias}.jsonl.gz", "rt", encoding="utf-8") as lines:
            asked = questions([json.loads(line) for line in lines])
        if per_task is not None and per_task < len(asked):
            asks = hashlib.sha256(canonical({k: v for k, v in config.items() if k not in SCORING_FIELDS})).hexdigest()
            asked = [asked[k] for k in sorted(random.Random(f"{seed}:{asks}").sample(range(len(asked)), per_task))]
        with open(work / "tasks" / f"{alias}.jsonl", "w", encoding="utf-8") as f:
            for label, group, unasked in asked:
                # As MCAccuracy does, a label that is not a number points to the first choice.
                label = 0 if label is None or isinstance(label, (str, list)) else label
                if READS[metric] in ("label", "first"):
                    at = label if READS[metric] == "label" else 0
                    record = {"label": 0, "idx": at, "pairs": [pair(group[at])]}
                else:
                    record = {"label": label, "pairs": [pair(r) for r in group]}
                if READS[metric] == "uncond":
                    if unasked is None:
                        raise ValueError(f"{alias} is scored without the questions but has no such requests")
                    record["uncond"] = [pair(r) for r in unasked]
                f.write(json.dumps({"doc_id": group[0]["doc_id"], **record}, ensure_ascii=False) + "\n")
    (work / "suites.json").write_text(json.dumps(kept, indent=1))
    (work / "tasks.json").write_text(json.dumps(metrics, indent=1))
    share = {"per_task": per_task, "seed": seed} if per_task is not None else {}
    return seal(work, "selection", requests=requests.name, olmes=made["olmes"], suites=sorted(suites), **share)


def reserve(requests, selection, out, per_task, seed=1):
    """A selection of questions that `selection` does not hold, from the requests that it was made from: of each of
    its tasks, a fixed random `per_task` of the task's other questions (all of them, if they are fewer), the same for
    tasks that ask the same. A task without another question is left out, and so is a group that then holds none."""
    made = check(selection)
    if made["requests"] != Path(requests).name:
        raise ValueError(f"{selection} was made from {made['requests']}")
    selection, out = Path(selection), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = json.loads((selection / "suites.json").read_text())
    metrics = json.loads((selection / "tasks.json").read_text())
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "tasks").mkdir()
    with tempfile.TemporaryDirectory(dir=out) as staging:
        whole = select(requests, staging, made["suites"])  # every question, as a selection holds them
        for alias in sorted(metrics):
            with open(selection / "tasks" / f"{alias}.jsonl", encoding="utf-8") as lines:
                taken = {json.loads(line)["doc_id"] for line in lines}
            with open(whole / "tasks" / f"{alias}.jsonl", encoding="utf-8") as lines:
                others = [(json.loads(line)["doc_id"], line) for line in lines]
            others = [(doc_id, line) for doc_id, line in others if doc_id not in taken]
            if per_task < len(others):  # chosen by the questions that are left, which tasks that ask the same share
                left = hashlib.sha256(json.dumps([doc_id for doc_id, _ in others]).encode()).hexdigest()
                others = [others[k] for k in sorted(random.Random(f"{seed}:{left}").sample(range(len(others)),
                                                                                       per_task))]
            if others:
                (work / "tasks" / f"{alias}.jsonl").write_text("".join(line for _, line in others), encoding="utf-8")
    kept = {alias: metric for alias, metric in metrics.items() if (work / "tasks" / f"{alias}.jsonl").exists()}

    def holds(name):
        return any(holds(member) for member in groups[name]) if name in groups else name in kept

    pruned = {name: [member for member in members if holds(member)] for name, members in groups.items() if holds(name)}
    (work / "suites.json").write_text(json.dumps(pruned, indent=1))
    (work / "tasks.json").write_text(json.dumps(kept, indent=1))
    return seal(work, "selection", requests=made["requests"], olmes=made["olmes"], suites=made["suites"],
                reserve_of=selection.name, per_task=per_task, seed=seed)


def is_reserve(selection):
    """Whether a selection was made by `reserve`: of the questions that another one does not hold."""
    return "reserve_of" in json.loads((Path(selection) / "manifest.json").read_text())


def within(whole, part):
    """Where each question of the selection `part` stands among the questions of the selection `whole`, for every
    task that both hold: {task: [index in the whole's task]}. A result on the whole, question by question, then
    scores the part: its questions' log-likelihoods are the whole's at these indices."""
    _, _, larger = read_selection(whole)
    _, _, smaller = read_selection(part)
    found = {}
    for alias in sorted(set(larger) & set(smaller)):
        index = {question["doc_id"]: at for at, question in enumerate(larger[alias])}
        missing = [question["doc_id"] for question in smaller[alias] if question["doc_id"] not in index]
        if missing:
            raise ValueError(f"{alias}: {whole} does not hold {len(missing)} questions of {part}")
        found[alias] = [index[question["doc_id"]] for question in smaller[alias]]
    return found


def read_selection(selection):
    """A selection's suites, its tasks' metrics and its tasks' questions, in the order that `pairs` lists them."""
    selection = Path(selection)
    groups = json.loads((selection / "suites.json").read_text())
    metrics = json.loads((selection / "tasks.json").read_text())
    tasks = {}
    for alias in sorted(metrics):
        with open(selection / "tasks" / f"{alias}.jsonl", encoding="utf-8") as lines:
            tasks[alias] = [json.loads(line) for line in lines]
    return groups, metrics, tasks


def pairs(tasks):
    """Every (context, continuation) that a selection's tasks read, in the order that rows and score take them."""
    return [p for alias in sorted(tasks) for question in tasks[alias]
            for p in question["pairs"] + question.get("uncond", [])]


def encode_rows(encode, bos, pairs, length=LENGTH):
    """Each (context, continuation) as OLMES's Hugging Face wrapper gives it to a model, with `encode` a function from
    texts to their tokens without special tokens: the context's trailing whitespace moves to the continuation, the
    continuation's tokens are what encoding both together adds after the context's own, the row begins with `bos`
    and keeps at most `length` + 1 tokens from its end. Returns each row's tokens and the number of its last tokens
    that are the continuation's."""
    contexts, wholes = [], []
    for context, continuation in pairs:
        if not context:
            raise ValueError("OLMES's wrapper reads a continuation without a context differently, and no task asks one")
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        contexts.append(context)
        wholes.append(context + continuation)
    rows = []
    # A few thousand at a time: a tokenizer's encodings of all of them at once can take tens of GB.
    for at in range(0, len(contexts), 2048):
        for context, whole in zip(encode(contexts[at:at + 2048]), encode(wholes[at:at + 2048])):
            answer = whole[len(context):]
            if not 0 < len(answer) <= length:
                raise ValueError(f"a continuation of {len(answer)} tokens, which OLMES's wrapper refuses")
            rows.append(([bos, *context, *answer][-(length + 1):], len(answer)))
    return rows


def tokenizer_encode(tokenizer_json):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    return lambda texts: [encoding.ids for encoding in tokenizer.encode_batch(texts, add_special_tokens=False)]


def rows(selection, out, tokenizer_json, bos, length=LENGTH):
    """A rows artifact in the directory `out`: encode_rows of a selection's pairs in a tokenizer, which `score` takes
    in their place when the tokenizer's contents are the same."""
    import numpy as np

    check(selection)
    _, _, tasks = read_selection(selection)
    made = encode_rows(tokenizer_encode(tokenizer_json), bos, pairs(tasks), length)
    tokens = np.concatenate([np.asarray(row, dtype=np.int64) for row, _ in made])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    with open(work / "rows.npz", "wb") as f:
        np.savez(f, tokens=tokens.astype(np.uint16 if tokens.max() < 1 << 16 else np.int32),
                 lengths=np.array([len(row) for row, _ in made]), answers=np.array([answer for _, answer in made]))
    return seal(work, "rows", selection=Path(selection).name, tokenizer=file_sha256(tokenizer_json), bos=bos,
                length=length)


def read_rows(rows_artifact, selection, tokenizer_json):
    """A rows artifact's rows, each as a view of one array, after checking that they encode this selection in this
    tokenizer."""
    import numpy as np

    made = check(rows_artifact)
    if made["selection"] != Path(selection).name or made["tokenizer"] != file_sha256(tokenizer_json):
        raise ValueError(f"{rows_artifact} encodes another selection or another tokenizer")
    with np.load(Path(rows_artifact) / "rows.npz") as kept:
        tokens, lengths, answers = kept["tokens"].astype(np.int64), kept["lengths"], kept["answers"]
    ends = np.cumsum(lengths)
    return [(tokens[end - length:end], int(answer)) for end, length, answer in zip(ends, lengths, answers)]


def windows(length, answer, context):
    """The windows in which a model of `context` tokens reads a row of `length` tokens for the predictions at its
    last `answer` positions, as (start, end, first): it reads the tokens from start to end and gives the predictions
    from first to end. A row within the context is read whole. A longer row is read in windows of the context that
    end at the row's end and step back by half of it, each position taking its prediction from the window that ends
    nearest after it, so that every position sees at least half a context before it; the last half of the row comes
    from the window that ends with it, as the wrapper's own cut to the context would give it."""
    if length <= context:
        return [(0, length, length - answer)]
    step, needed = context // 2, length - answer
    found, end = [], length
    while end > needed:
        start = max(end - context, 0)
        first = end - step if start > 0 else 0
        found.append((start, end, max(first, needed)))
        end = first
    return found


def reader(logits, context, pad, device, batch_tokens=65536, most_rows=None):
    """A function that reads rows with a model: for each row, the log-likelihood in nats of its continuation (the sum
    over its tokens of the log-softmax of the logits that predict them) and whether each of its tokens has the
    highest logit. `logits` takes a batch of token rows on `device` and gives the model's logits at every position;
    the windows of all the rows are read longest first, in batches of about batch_tokens tokens and at most most_rows
    rows, padded at the end with `pad`."""
    import torch

    @torch.inference_mode()
    def read(rows):
        cut = sorted(((index, *window) for index, (tokens, answer) in enumerate(rows)
                      for window in windows(len(tokens) - 1, answer, context)),
                     key=lambda window: window[2] - window[1], reverse=True)
        sums = torch.zeros(len(rows), dtype=torch.float64, device=device)
        greedy = torch.ones(len(rows), dtype=torch.bool, device=device)
        at = 0
        while at < len(cut):
            longest = cut[at][2] - cut[at][1]
            batch = cut[at:at + max(1, min(batch_tokens // longest, most_rows or len(cut)))]
            at += len(batch)
            inputs = torch.full((len(batch), longest), pad, dtype=torch.long)
            lines, positions, targets, owners = [], [], [], []
            for line, (index, start, end, first) in enumerate(batch):
                tokens = torch.as_tensor(rows[index][0])
                inputs[line, :end - start] = tokens[start:end]
                lines.append(torch.full((end - first,), line))
                positions.append(torch.arange(first - start, end - start))
                targets.append(tokens[first + 1:end + 1])
                owners.append(torch.full((end - first,), index))
            picked = logits(inputs.to(device))[torch.cat(lines).to(device), torch.cat(positions).to(device)].float()
            target, owner = torch.cat(targets).to(device), torch.cat(owners).to(device)
            sums.index_add_(0, owner, picked.log_softmax(-1).gather(1, target[:, None])[:, 0].double())
            greedy[owner[picked.argmax(-1) != target]] = False
        return sums.tolist(), greedy.tolist()

    return read


def question_values(metric, questions, totals, greedy, answers):
    """Each question's value under its task's primary metric, as OLMES's MCAccuracy computes it before it averages
    them, from the log-likelihoods, the greedy flags and the continuations' tokens of the questions' rows, in their
    order. Of equal scores the first choice wins."""
    values, at = [], 0
    for question in questions:
        n = len(question["pairs"])
        own, mine = totals[at:at + n], range(at, at + n)
        if metric == "bits_per_byte_corr":
            values.append(-LOG_2_OF_E * (own[0] / max(len(question["pairs"][0][1].encode("utf-8")), 1)))
        elif metric == "greedy_acc":
            values.append(float(greedy[at]))
        elif metric == "acc_per_byte":
            bits = [-LOG_2_OF_E * (total / max(len(c.encode("utf-8")), 1)) for total, (_, c) in zip(own, question["pairs"])]
            values.append(float(bits.index(min(bits)) == question["label"]))
        else:
            by = {"acc_raw": own,
                  "acc_per_token": [total / max(answers[k], 1) for total, k in zip(own, mine)],
                  "acc_per_char": [total / max(len(c), 1) for total, (_, c) in zip(own, question["pairs"])],
                  "acc_uncond": [total - without for total, without in zip(own, totals[at + n:at + 2 * n])]}[metric]
            values.append(float(by.index(max(by)) == question["label"]))
        at += n + len(question.get("uncond", []))
    return values


def task_score(metric, questions, totals, greedy, answers):
    """A task's score under its primary metric: the mean of its questions' values."""
    values = question_values(metric, questions, totals, greedy, answers)
    return sum(values) / len(values)


def suite_scores(groups, scores):
    """Each suite's score from its tasks' scores: a suite's or a group's score is the mean of its members'."""

    def mean_of(name):
        if name not in groups:
            return scores[name]
        members = [mean_of(member) for member in groups[name]]
        return sum(members) / len(members)

    return {name: mean_of(name) for name in groups if not any(name in members for members in groups.values())}


def values(selection, read, made_rows):
    """Each task's questions' values from a selection, its rows (encode_rows of its pairs, or read_rows of a rows
    artifact) and a reader: {task: [value]}, the questions in the selection's order."""
    _, metrics, tasks = read_selection(selection)
    totals, greedy = read(made_rows)
    answers = [answer for _, answer in made_rows]
    found, at = {}, 0
    for alias, asked in tasks.items():
        size = sum(len(q["pairs"]) + len(q.get("uncond", [])) for q in asked)
        found[alias] = question_values(metrics[alias], asked, totals[at:at + size], greedy[at:at + size],
                                       answers[at:at + size])
        at += size
    return found


def score(selection, read, made_rows):
    """Each task's score and each suite's from a selection, its rows (encode_rows of its pairs, or read_rows of a rows
    artifact) and a reader."""
    groups = json.loads((Path(selection) / "suites.json").read_text())
    scores = {alias: sum(found) / len(found) for alias, found in values(selection, read, made_rows).items()}
    return scores, suite_scores(groups, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    chosen = commands.add_parser("select")
    chosen.add_argument("requests")
    chosen.add_argument("out")
    chosen.add_argument("suites", nargs="+")
    chosen.add_argument("--per-task", type=int)
    chosen.add_argument("--seed", type=int, default=0)
    encoded = commands.add_parser("rows")
    encoded.add_argument("selection")
    encoded.add_argument("out")
    encoded.add_argument("tokenizer")
    encoded.add_argument("bos", type=int)
    aside = commands.add_parser("reserve")
    aside.add_argument("requests")
    aside.add_argument("selection")
    aside.add_argument("out")
    aside.add_argument("--per-task", type=int, required=True)
    aside.add_argument("--seed", type=int, default=1)
    checked = commands.add_parser("check")
    checked.add_argument("artifacts", nargs="+")
    args = parser.parse_args()
    if args.command == "select":
        print(select(args.requests, args.out, args.suites, args.per_task, args.seed))
    elif args.command == "reserve":
        print(reserve(args.requests, args.selection, args.out, args.per_task, args.seed))
    elif args.command == "rows":
        print(rows(args.selection, args.out, args.tokenizer, args.bos))
    else:
        for artifact in args.artifacts:
            print(artifact, check(artifact)["kind"])


if __name__ == "__main__":
    main()
