"""Following instructions, as IFEval and M-IFEval ask and score it.

A model writes a reply to each prompt, and the authors' code tells whether the reply follows every instruction that
the prompt gives (strict) or would after a light cleaning (loose). M-IFEval's code (lightblue-tech/M-IFEval) holds
IFEval's English prompts and checks beside its Japanese ones, so one commit of it asks and scores both languages:

- `prompts(language, home)`: the prompts of a language, as the authors give them: key, prompt, instruction ids and
  their arguments;
- `score(language, replies, work)`: each prompt's results at both levels, from the authors' evaluation_main run
  unchanged in an environment of their pins, and the share of prompts whose every instruction is followed;
- `verify(work)`: the authors' published results of one model repeated from its published replies, which shows that
  the environment scores as theirs did. Seven Japanese prompts of the input file say N文 (sentences) where the
  published replies were asked N行 (lines), as llm-jp-eval still asks them, with the same instructions to check: the
  prompts are asked as the input file says, which its checks count, and `verify` asks the seven as their replies were.

The authors' checks draw a random letter when the one to count is not a letter (an English prompt counts "!"), and
langdetect draws its own: both are seeded, so that a reply scores the same each time. The authors' published result
of that English prompt was drawn otherwise, so `verify` repeats 540 of the 541 at the strict level.

A project writes the replies itself, each prompt the user's message of a conversation in its model's format, greedily
up to MAX_NEW_TOKENS, as the authors asked the models whose replies they publish (temperature 0, 2048 tokens).
"""
import json
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/lightblue-tech/M-IFEval.git"
COMMIT = "10b874dbf8c4ca25ed5d876e1d3226e7f3f9d0dc"
LANGUAGES = ("en", "ja")
LEVELS = ("strict", "loose")
MAX_NEW_TOKENS = 2048
# The authors' pins of what evaluation_main imports; the rest of their requirements serve to write replies.
PACKAGES = ("absl-py==1.4.0", "immutabledict==4.2.1", "janome==0.5.0", "ja_sentence_segmenter==0.0.2",
            "langdetect==1.0.9", "nltk==3.9.1", "spacy==3.7.5")
# The models that their README downloads, which their instruction modules load on import.
MODELS = tuple(f"https://github.com/explosion/spacy-models/releases/download/{name}-3.7.0/{name}-3.7.0-py3-none-any.whl"
               for name in ("es_core_news_sm", "xx_sent_ud_sm"))
# The model whose published replies and results `verify` repeats.
PUBLISHED = "input_response_data_Qwen__Qwen2.5-0.5B-Instruct-GPTQ-Int4"


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def code(work):
    """The authors' code at COMMIT, in work/code."""
    home = Path(work) / "code"
    if not (home / "evaluation_main.py").exists():
        home.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--quiet", cwd=home)
        run("git", "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT, cwd=home)
        run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=home)
    return home


def environment(work):
    """The Python of an environment with the authors' pins, in work/environment."""
    python = Path(work) / "environment" / "bin" / "python"
    if not python.exists():
        run("uv", "venv", "--quiet", "--python", "3.12", str(python.parent.parent))
        run("uv", "pip", "install", "--quiet", "--python", str(python), *PACKAGES, *MODELS)
    return python


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prompts(language, home):
    """The authors' prompts of a language, in their order."""
    if language not in LANGUAGES:
        raise ValueError(f"no prompts of {language}")
    return read_lines(Path(home) / "data" / f"{language}_input_data.jsonl")


def evaluated(asked, replies, directory, home, python):
    """The authors' results of each level for replies ({prompt: reply}) to prompts (as `prompts` gives them)."""
    if set(replies) != {row["prompt"] for row in asked}:
        raise ValueError("the replies do not answer each of the prompts once")
    # Each instruction's id names its language ("ja:..."), by which the authors' code picks its checks.
    write_lines(directory / "input.jsonl", asked)
    write_lines(directory / "responses.jsonl", [{"prompt": row["prompt"], "response": replies[row["prompt"]]}
                                                for row in asked])
    flags = [f"--input_data={directory / 'input.jsonl'}", f"--input_response_data={directory / 'responses.jsonl'}",
             f"--output_dir={directory}"]
    seeded = ("import random, sys, langdetect; random.seed(0); langdetect.DetectorFactory.seed = 0; "
              f"sys.argv = ['evaluation_main', *{flags!r}]; "
              "import evaluation_main; from absl import app; app.run(evaluation_main.main)")
    run(str(python), "-c", seeded, cwd=home)
    return {level: read_lines(directory / f"eval_results_{level}.jsonl") for level in LEVELS}


def score(language, replies, work):
    """A language's results for replies ({prompt: reply}): each level's share of prompts whose every instruction is
    followed, and each prompt's results."""
    home, python = code(work), environment(work)
    results = evaluated(prompts(language, home), replies, Path(work) / "scored" / language, home, python)
    shares = {level: sum(row["follow_all_instructions"] for row in rows) / len(rows) for level, rows in results.items()}
    return {"language": language, "commit": COMMIT, "prompts": len(results["strict"]), **shares, "results": results}


def as_replied(asked, replies):
    """The prompts as the replies were asked: a prompt that no reply answers takes the wording of the one that answers
    it with 行 for 文, and the same text otherwise."""
    unanswered = set(replies) - {row["prompt"] for row in asked}

    def earlier(prompt):
        if prompt in replies:
            return prompt
        found = [other for other in unanswered if len(other) == len(prompt) and all(
            a == b or (a, b) == ("文", "行") for a, b in zip(prompt, other))]
        if len(found) != 1:
            raise ValueError(f"no reply answers {prompt[:40]!r}")
        return found[0]

    return [{**row, "prompt": earlier(row["prompt"])} for row in asked]


def verify(work):
    """How many of the authors' published results of each language and level the environment repeats from their
    published replies, of how many: {language: {level: (repeated, prompts)}}."""
    home, python = code(work), environment(work)
    found = {}
    for language in LANGUAGES:
        published = f"{language}_{PUBLISHED}"
        replies = {row["prompt"]: row["response"] for row in read_lines(home / "data" / f"{published}.jsonl")}
        asked = as_replied(prompts(language, home), replies)
        ours = evaluated(asked, replies, Path(work) / "verified" / language, home, python)
        found[language] = {}
        for level in LEVELS:
            theirs = {row["prompt"]: row["follow_instruction_list"]
                      for row in read_lines(home / "evaluations" / published / f"eval_results_{level}.jsonl")}
            repeated = sum(theirs.get(row["prompt"]) == row["follow_instruction_list"] for row in ours[level])
            found[language][level] = (repeated, len(ours[level]))
    return found
