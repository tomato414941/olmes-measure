"""Build every request of some OLMES suites with OLMES's own runner, without a model, as a requests artifact that
olmes_measure.py selects from. It runs where OLMES is installed at REVISION, as olmes.txt installs it.

The artifact holds, for each task of the suites, OLMES's requests as its runner writes them (requests/ALIAS.jsonl.gz),
the task's configuration as the runner resolves it (tasks.json), and how the suites average their tasks (suites.json).
Requests do not depend on a model; OLMES reads all of a task's few-shot source for every question, so building takes
time that grows with the square of a task's size, and the tasks are shared among processes. Once the tasks' data are
in the Hugging Face cache, HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 build without asking the Hub, which refuses a
machine that asks too often, and in a few minutes.

    python olmes_build.py OUT SUITE... [--workers N]
"""

import argparse
import glob
import gzip
import json
import multiprocessing
import os
import shlex
import shutil
import tempfile
from importlib.metadata import distribution
from pathlib import Path

from olmes_measure import seal

REVISION = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"


def installed_revision():
    """The commit of the OLMES that is installed, from the archive's URL that pip recorded."""
    url = json.loads(distribution("ai2-olmes").read_text("direct_url.json"))["url"]
    return next((part for part in url.replace(".zip", "").split("/") if len(part) == 40), url)


def read_datasets_4_features():
    """Some tasks' data on the Hub were saved by `datasets` 4, which names list features `List`. OLMES pins `datasets` 3,
    which calls the same feature `Sequence`; with that name it reads the same examples."""
    from datasets.features import features

    features._FEATURE_TYPES.setdefault("List", features.Sequence)


def runner_arguments(suites, output_dir):
    """The arguments of OLMES's runner for the suites without a model, as its launcher makes them from their names."""
    from oe_eval import launch, run_eval

    options = ["--task", *suites, "--model", "none", "--output-dir", str(output_dir), "--save-raw-requests", "true",
               "--dry-run"]
    command = launch.launch_eval(vars(launch._parser.parse_args(options)))["command"]
    words = shlex.split(command)
    assert words[:3] == ["python", "-m", "oe_eval.run_eval"], command
    return vars(run_eval._parser.parse_args(words[3:]))


def task_alias(task_config):
    """The name that OLMES's launcher gives a task of a suite, which its aggregation reads."""
    return task_config.get("metadata", {}).get("alias") or task_config["task_name"]


def suite_groups(suites):
    """The groups that OLMES's aggregation averages for the suites, the suites among them, each with its members: tasks,
    or groups, in the order of their names, since OLMES keeps some groups' members in a set, whose order changes from
    one run to the next. Every group takes the macro average of its members. A suite that a group holds and that
    OLMES's aggregation has no rule for, such as ARC's two tasks in the QA suites, gets no score from OLMES, so it
    takes the macro average too."""
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.tasks import aggregate_tasks

    groups = {}

    def add(name):
        group = aggregate_tasks.TASK_GROUPS.get(name)
        if group is None:
            group = {"tasks": TASK_SUITE_CONFIGS[name]["tasks"], "primary_metric": "macro"}
        if group["primary_metric"] != "macro":
            raise ValueError(f"{name} takes the {group['primary_metric']} average, which olmes_measure does not")
        groups[name] = sorted(group["tasks"])
        for member in groups[name]:
            if member in aggregate_tasks.TASK_GROUPS or member in TASK_SUITE_CONFIGS:
                add(member)

    for suite in suites:
        add(suite)
    return groups


def build_share(suites, tasks, output_dir):
    from oe_eval import run_eval

    read_datasets_4_features()
    arguments = runner_arguments(suites, output_dir)
    arguments["task"] = tasks
    run_eval.run_eval(arguments)


def build(out, suites, workers=5):
    """A requests artifact in the directory `out` with every request of the suites."""
    if installed_revision() != REVISION:
        raise RuntimeError(f"OLMES {installed_revision()} is installed, not {REVISION}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    tasks = runner_arguments(suites, work / "unused")["task"]
    shares = [(suites, tasks[k::workers], str(work / f"worker-{k}")) for k in range(workers) if tasks[k::workers]]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=build_share, args=share) for share in shares]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError("a process that builds requests failed")
    (work / "requests").mkdir()
    configs = {}
    for _, _, directory in shares:
        for metrics_path in sorted(glob.glob(os.path.join(directory, "*-metrics.json"))):
            config = json.load(open(metrics_path))["task_config"]
            alias = task_alias(config)
            configs[alias] = config
            with open(metrics_path.removesuffix("-metrics.json") + "-requests.jsonl", "rb") as source, \
                    open(work / "requests" / f"{alias}.jsonl.gz", "wb") as target, \
                    gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as packed:
                shutil.copyfileobj(source, packed)
        shutil.rmtree(directory)
    groups = suite_groups(suites)
    held = {member for members in groups.values() for member in members if member not in groups}
    if held != set(configs) or len(configs) != len(tasks):
        raise RuntimeError(f"built {len(configs)} of {len(tasks)} tasks, and the suites' groups hold {len(held)}")
    shutil.rmtree(work / "unused", ignore_errors=True)
    (work / "tasks.json").write_text(json.dumps(configs, indent=1, sort_keys=True))
    (work / "suites.json").write_text(json.dumps(groups, indent=1))
    return seal(work, "requests", olmes=REVISION, suites=sorted(suites))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("suites", nargs="+")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    print(build(args.out, args.suites, args.workers))


if __name__ == "__main__":
    main()
