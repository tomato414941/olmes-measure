"""Evaluation definitions: what a project measures by, under a name.

A definition (definitions/NAME.json) says
- the data that is measured, as the tools make it and as the artifacts that they make are named: OLMES's questions
  (olmes_measure.select: the suites that the categories count, or with `selected` those and others that a project
  measures besides; every question or so many of each task), Paloma's text (paloma.fetch: which split; paloma.part:
  how much of it), and what is kept aside from decisions, a reserve, if any;
- the text that nothing measured by it may learn (overlap.make: every question of the requests, Paloma's splits
  whole), by the artifact's name;
- the categories that a result is divided into, each the mean of its members: OLMES's tasks and groups, a named
  mean of some tasks, or Paloma's sources and domains; and
- how two results compare (`improvement`).

A definition is not edited once a result names it: a change is a new name. One that is `within` another measures a
part of the other's data and says only what differs, so that the two agree on everything else and a result by the
larger one can be compared on the part: OLMES's questions by olmes_measure.within, Paloma's text by reading the part.

    python -m language_model_evaluation.definition make NAME OUT REQUESTS   # the definition's artifacts, in OUT
    python -m language_model_evaluation.definition show NAME

`make` builds what OUT does not hold yet and checks every name against the definition's; fetching Paloma needs
HF_TOKEN (paloma.py).
"""
import argparse
import json
from pathlib import Path

import olmes_measure

from . import overlap, paloma

DEFINITIONS = Path(__file__).parent / "definitions"


def load(name):
    """The definition of a name, with what it takes from the one that it is within."""
    own = json.loads((DEFINITIONS / f"{name}.json").read_text(encoding="utf-8"))
    if "within" not in own:
        return {"name": name, **own}
    whole = load(own["within"])
    merged = {**whole, **own, "name": name}
    for section in ("olmes", "paloma"):
        merged[section] = {**whole[section], **own.get(section, {})}
    return merged


def members(definition):
    """Paloma's members of each category that is read: {category: [member]}, as paloma.fetch and paloma.part take
    them."""
    return {category: list(parts["paloma"]) for category, parts in definition["categories"].items() if "paloma" in parts}


def make(name, out, requests, download=paloma.download, listed=paloma.listed):
    """Every artifact that a definition (its name, or one already loaded) names, in the directory `out`, from an
    olmes-measure requests artifact and Paloma's files: {name: its directory}. One that `out` holds already is
    checked and kept; one that is made must come out under the definition's name."""
    definition, out, made = load(name) if isinstance(name, str) else name, Path(out), {}
    name = definition["name"]

    def have(expected, build):
        if (out / expected).exists():
            olmes_measure.check(out / expected)
        else:
            artifact = build()
            if artifact.name != expected:
                raise ValueError(f"{name} names {expected}, and what was made is {artifact.name}")
        made[expected] = out / expected
        return out / expected

    olmes, text, kinds = definition["olmes"], definition["paloma"], members(definition)
    if Path(requests).name != olmes["requests"]:
        raise ValueError(f"{name} is made from {olmes['requests']}")
    selection = have(olmes["selection"], lambda: olmes_measure.select(
        requests, out, olmes.get("selected", olmes["suites"]), olmes.get("per_task"), olmes.get("seed", 0)))
    if "reserve" in olmes:
        aside = olmes["reserve"]
        have(aside["selection"], lambda: olmes_measure.reserve(requests, selection, out, aside["per_task"],
                                                               aside["seed"]))
    fetch = lambda split: lambda: paloma.fetch(out, split, kinds, download, listed)
    whole = have(text["whole"], fetch(text["split"]))
    if text.get("part"):
        have(text["text"], lambda: paloma.part(whole, out, kinds, **text["part"]))
    if "reserve" in text:
        aside = text["reserve"]
        other = have(aside["whole"], fetch(aside["split"]))
        have(aside["text"], lambda: paloma.part(other, out, kinds, **text["part"]))
    splits = [have(artifact, fetch(split)) for split, artifact in definition["overlap"]["paloma"].items()]
    have(definition["overlap"]["artifact"], lambda: overlap.make(out, requests, splits))
    return made


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def answered(definition, groups, scores, partial=False):
    """The score of each category that is answered, from each OLMES task's score ({task: score}) and the groups in
    which a selection holds the suites' tasks. A member is a task, a group (the mean of its members, as OLMES averages
    them), or a named mean of some tasks ({name: [task]}). The categories count every task of the definition's suites
    once; a selection that is `partial`, as a reserve is, may lack tasks, and its categories are of the others."""

    def tasks(member):
        return list(*member.values()) if isinstance(member, dict) else olmes_measure.leaves(groups, member)

    def score(member):
        if isinstance(member, dict):
            return _mean(scores[task] for task in tasks(member) if task in scores)
        return _mean(score(inner) for inner in groups[member]) if member in groups else scores[member]

    sets = {category: parts["olmes"] for category, parts in definition["categories"].items() if "olmes" in parts}
    counted = [task for parts in sets.values() for member in parts for task in tasks(member)]
    suites = [task for suite in definition["olmes"]["suites"] if suite in groups
              for task in olmes_measure.leaves(groups, suite)]
    asked = {task for task in suites if task in scores}
    if len(set(counted)) != len(counted) or not asked <= set(counted):
        raise ValueError("the categories do not count every task of the suites once")
    if not partial and set(counted) != asked:
        raise ValueError("the categories count tasks that the selection does not hold")
    held = lambda member: any(task in scores for task in tasks(member))
    return {category: _mean(score(member) for member in parts if held(member)) for category, parts in sets.items()}


def read(definition, domains, partial=False):
    """The score of each category that is read, from each of Paloma's domains' bits per byte ({SOURCE/DOMAIN: BPB}):
    a member is its one domain or the mean of its source's, and a category the mean of its members. Every member is
    scored, unless the text is `partial`, as a reserve's may be: its categories are of the members that it holds."""

    def member(name):
        source, _, one = name.partition("/")
        found = [bits for key, bits in domains.items()
                 if key.partition("/")[0] == source and one in ("", key.partition("/")[2])]
        if not found and not partial:
            raise ValueError(f"no domain of {name} is scored")
        return sum(found) / len(found) if found else None

    scores = {}
    for category, names in members(definition).items():
        found = [bits for bits in map(member, names) if bits is not None]
        if not found:
            raise ValueError(f"no member of {category} is scored")
        scores[category] = sum(found) / len(found)
    return scores


def categories(definition):
    """The categories' names, in the definition's order."""
    return tuple(definition["categories"])


def improvement(definition, scores, standing):
    """By how much `scores` improve on `standing` ({category: score}, of every category): under the rule
    `mean_rate`, the mean of the categories' rates of improvement, each counting equally. Positive: `scores` are the
    better."""
    if definition["compare"] != "mean_rate":
        raise ValueError(f"no rule {definition['compare']}")
    return _mean(1 - scores[category] / standing[category] for category in categories(definition))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("make")
    made.add_argument("name")
    made.add_argument("out")
    made.add_argument("requests")
    shown = commands.add_parser("show")
    shown.add_argument("name")
    args = parser.parse_args()
    if args.command == "make":
        for artifact in make(args.name, args.out, args.requests):
            print(artifact)
    else:
        print(json.dumps(load(args.name), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
