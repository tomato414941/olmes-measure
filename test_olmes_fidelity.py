"""Checks olmes_measure against OLMES itself where OLMES is installed (olmes.txt): every task of a requests artifact
scored from the same random responses by OLMES's own metrics and by olmes_measure, the suites averaged by OLMES's
aggregation and by olmes_measure, and rows made by OLMES's wrapper and by olmes_measure.

    OLMES_MEASURE_REQUESTS=ARTIFACT python -m unittest test_olmes_fidelity
"""

import functools
import gzip
import importlib.util
import json
import os
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import olmes_measure as om

HAS_OLMES = all(importlib.util.find_spec(name) is not None for name in ("oe_eval", "transformers", "tokenizers"))
REQUESTS = os.environ.get("OLMES_MEASURE_REQUESTS")


def olmes_metrics(config):
    """The metrics that OLMES's runner scores a task with: the task's own, or MCAccuracy alone for bits per byte."""
    from oe_eval import run_eval
    from oe_eval.metrics.metric import MCAccuracy

    task = run_eval.load_task(config)
    task.make_metrics()
    if config["primary_metric"] == "bits_per_byte_corr":
        return [MCAccuracy(**(config.get("metric_kwargs") or {}))]
    return task._metrics


@unittest.skipUnless(HAS_OLMES and REQUESTS, "needs OLMES and a requests artifact in OLMES_MEASURE_REQUESTS")
class FidelityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = Path(REQUESTS)
        made = om.check(cls.requests)
        cls.configs = json.loads((cls.requests / "tasks.json").read_text())
        cls.groups = json.loads((cls.requests / "suites.json").read_text())
        cls.directory = tempfile.TemporaryDirectory()
        cls.selection = om.select(cls.requests, cls.directory.name, made["suites"])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_scores_every_task_as_olmes_scores_it_from_every_request(self):
        _, metrics, tasks = om.read_selection(self.selection)
        self.assertEqual(set(tasks), set(self.configs))
        rng = random.Random(0)
        for alias, config in sorted(self.configs.items()):
            with gzip.open(self.requests / "requests" / f"{alias}.jsonl.gz", "rt", encoding="utf-8") as lines:
                requests = [json.loads(line) for line in lines]
            responses = {}
            for request in requests:
                key = (request["doc_id"], request["idx"])
                tied = request["idx"] > 0 and rng.random() < 0.2  # equal to the choice before, now and then
                responses[key] = (dict(responses[(key[0], key[1] - 1)]) if tied else
                                  {"sum_logits": -rng.uniform(1, 30), "num_tokens": rng.randint(1, 9),
                                   "is_greedy": rng.random() < 0.5})
            expected = None
            for metric in olmes_metrics(config):
                metric.compute_for_docs([{**r, "model_resps": dict(responses[(r["doc_id"], r["idx"])])} for r in requests])
                expected = metric.aggregate_to_task(primary_metric=config["primary_metric"]).get("primary_score", expected)
            totals, greedy, answers = [], [], []
            for question in tasks[alias]:
                keys = ([(question["doc_id"], question["idx"])] if "idx" in question else
                        [(question["doc_id"], k) for k in range(len(question["pairs"]))])
                keys += [(question["doc_id"] + om.UNCONDITIONED, k) for k in range(len(question.get("uncond", [])))]
                for key in keys:
                    totals.append(responses[key]["sum_logits"])
                    greedy.append(responses[key]["is_greedy"])
                    answers.append(responses[key]["num_tokens"])
            ours = om.task_score(metrics[alias], tasks[alias], totals, greedy, answers)
            # The means may add in another order, which changes only the last bits.
            self.assertAlmostEqual(ours, expected, delta=1e-12 * max(1.0, abs(expected)), msg=alias)

    def test_averages_the_suites_as_olmes_aggregates_them(self):
        from oe_eval.tasks import aggregate_tasks
        from oe_eval.tasks.aggregate_tasks import add_aggregate_tasks

        rng = random.Random(0)
        scores = {alias: rng.uniform(0.2, 3.0) for alias in self.configs}
        entries = [{"task_name": alias, "task_idx": k, "num_instances": rng.randint(10, 1000),
                    "metrics": {"primary_score": score}, "task_config": {"metadata": {"alias": alias}},
                    "model_hash": "", "model_config": {}, "compute_config": {}, "processing_time": 0,
                    "current_date": "", "beaker_info": {}} for k, (alias, score) in enumerate(scores.items())]
        # The groups that OLMES has no rule for take the macro average here, as olmes_build.py gives them.
        missing = {name: {"tasks": members, "expected_count": len(members), "primary_metric": "macro"}
                   for name, members in self.groups.items() if name not in aggregate_tasks.TASK_GROUPS}
        with mock.patch.object(aggregate_tasks, "TASK_GROUPS", {**missing, **aggregate_tasks.TASK_GROUPS}):
            aggregated = {entry["task_name"]: entry["metrics"]["primary_score"] for entry in add_aggregate_tasks(entries)}
        for suite, value in om.suite_scores(self.groups, scores).items():
            self.assertAlmostEqual(value, aggregated[suite], places=12, msg=suite)


@unittest.skipUnless(HAS_OLMES, "needs OLMES")
class RowsFidelityTest(unittest.TestCase):
    def test_makes_the_rows_that_olmes_wrapper_makes(self):
        import transformers
        from lm_eval.api.model import TemplateLM
        from oe_eval.models.eleuther_huggingface import HFLM_Verbose
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        texts = ["The cat sat on the mat. Ünïcödé text, with numbers 12345.", "Another document; café au lait."] * 20
        tokenizer = Tokenizer(models.BPE())
        # Without a space added before the text, which transformers' wrapping of the tokenizer would not add.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=300, special_tokens=["<|bos|>"]))
        bos = tokenizer.token_to_id("<|bos|>")
        fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<|bos|>")
        wrapper = types.SimpleNamespace(tokenizer=fast, add_bos_token=True, AUTO_MODEL_CLASS=transformers.AutoModelForCausalLM)
        wrapper.tok_encode = functools.partial(HFLM_Verbose.tok_encode, wrapper)
        pairs = [(texts[k] + "  ", texts[k + 1][:k + 3]) for k in range(10)] + [("Question: what?\nAnswer:", " café")]
        encode = lambda batch: [e.ids for e in tokenizer.encode_batch(batch, add_special_tokens=False)]
        for (tokens, answer), (context, continuation) in zip(om.encode_rows(encode, bos, pairs, length=16), pairs):
            before, after = TemplateLM._encode_pair(wrapper, context, continuation)
            self.assertEqual(tokens, (before + after)[-17:])
            self.assertEqual(answer, len(after))


if __name__ == "__main__":
    unittest.main()
