import gzip
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import olmes_measure as om

HAS_TORCH = importlib.util.find_spec("torch") is not None


def request(doc_id, idx, label, continuation, context="Q:", native_id=None):
    return {"doc_id": doc_id, "native_id": doc_id % om.UNCONDITIONED if native_id is None else native_id, "idx": idx,
            "label": label, "request": {"context": context, "continuation": continuation}}


def multiple_choice(labels, choices=3, unconditioned=False):
    """A task's requests as OLMES writes them: each question's choices, then the same without the questions."""
    asked = [request(doc_id, idx, label, f" choice {doc_id}-{idx}") for doc_id, label in enumerate(labels)
             for idx in range(choices)]
    unasked = [{**r, "doc_id": r["doc_id"] + om.UNCONDITIONED,
                "request": {"context": "Answer:", "continuation": r["request"]["continuation"]}} for r in asked]
    return asked + (unasked if unconditioned else [])


def make_requests(directory, tasks, groups):
    """A requests artifact in directory from {alias: (config, requests)} and the suites' groups."""
    work = Path(tempfile.mkdtemp(dir=directory))
    (work / "requests").mkdir()
    for alias, (_, requests) in tasks.items():
        with gzip.open(work / "requests" / f"{alias}.jsonl.gz", "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in requests)
    (work / "tasks.json").write_text(json.dumps({alias: config for alias, (config, _) in tasks.items()}))
    (work / "suites.json").write_text(json.dumps(groups))
    return om.seal(work, "requests", olmes="test", suites=sorted(groups))


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def made(self, text="a"):
        work = Path(tempfile.mkdtemp(dir=self.directory))
        (work / "inner").mkdir()
        (work / "inner" / "file.txt").write_text(text)
        return om.seal(work, "thing", source="test")


class SealTest(ArtifactTest):
    def test_names_an_artifact_by_its_kind_and_its_manifest(self):
        artifact = self.made()
        self.assertRegex(artifact.name, r"^thing-[0-9a-f]{16}$")
        self.assertEqual(om.check(artifact)["files"], {"inner/file.txt": om.file_sha256(artifact / "inner/file.txt")})

    def test_names_equal_contents_equally_and_keeps_the_one_made_first(self):
        first, second = self.made(), self.made()
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.directory.iterdir())), 1)
        self.assertNotEqual(self.made("b"), first)

    def test_finds_a_file_changed_or_added_since_it_was_sealed(self):
        artifact = self.made()
        (artifact / "inner" / "file.txt").write_text("b")
        with self.assertRaises(ValueError):
            om.check(artifact)
        artifact = self.made("c")
        (artifact / "extra.txt").write_text("")
        with self.assertRaises(ValueError):
            om.check(artifact)


class QuestionsTest(unittest.TestCase):
    def test_groups_each_questions_choices_with_the_same_choices_without_the_question(self):
        grouped = om.questions(multiple_choice([2, 0], unconditioned=True))
        self.assertEqual([label for label, _, _ in grouped], [2, 0])
        label, asked, unasked = grouped[1]
        self.assertEqual([r["request"]["continuation"] for r in asked], [" choice 1-0", " choice 1-1", " choice 1-2"])
        self.assertEqual([r["doc_id"] for r in unasked], [1 + om.UNCONDITIONED] * 3)

    def test_refuses_requests_without_their_question_in_another_order(self):
        requests = multiple_choice([1], unconditioned=True)
        requests[3:] = requests[3:][::-1]
        with self.assertRaises(ValueError):
            om.questions(requests)

    def test_refuses_a_questions_choices_out_of_the_order_of_their_indices(self):
        requests = multiple_choice([1])
        requests[0], requests[1] = requests[1], requests[0]
        with self.assertRaises(ValueError):
            om.questions(requests)


class SelectTest(ArtifactTest):
    def setUp(self):
        super().setUp()
        asks = {"task_name": "quiz", "split": "test", "num_shots": 5}
        self.tasks = {
            "quiz:bpb": ({**asks, "primary_metric": "bits_per_byte_corr", "compute_gold_bpb": True},
                         multiple_choice([2, 0, 1, 2, 0, 1], unconditioned=True)),
            "quiz:rc": ({**asks, "primary_metric": "acc_per_char", "metadata": {"alias": "quiz:rc"}},
                        multiple_choice([2, 0, 1, 2, 0, 1], unconditioned=True)),
            "pick:rc": ({"task_name": "pick", "primary_metric": "acc_uncond"},
                        multiple_choice([1, 0], choices=2, unconditioned=True)),
            "words:bpb": ({"task_name": "words", "primary_metric": "bits_per_byte_corr"},
                          [request(0, 0, "x", " a"), request(1, 0, None, " b"), request(2, 0, ["y"], " c")]),
            "last": ({"task_name": "last", "primary_metric": "greedy_acc"},
                     [request(0, 0, 1, " end"), request(0, 1, 1, " other")]),
        }
        self.groups = {"bpb": ["inner", "words:bpb"], "inner": ["quiz:bpb"], "rc": ["quiz:rc", "pick:rc", "last"]}
        self.requests = make_requests(self.directory, self.tasks, self.groups)

    def selected(self, suites, **share):
        selection = om.select(self.requests, self.directory / "selections", suites, **share)
        return selection, om.read_selection(selection)

    def test_keeps_only_the_answer_at_each_label_for_bits_per_byte_and_the_first_for_a_label_not_a_number(self):
        _, (groups, metrics, tasks) = self.selected(["bpb"])
        self.assertEqual(groups, {"bpb": ["inner", "words:bpb"], "inner": ["quiz:bpb"]})
        self.assertEqual(metrics, {"quiz:bpb": "bits_per_byte_corr", "words:bpb": "bits_per_byte_corr"})
        self.assertEqual([q["pairs"] for q in tasks["quiz:bpb"][:2]], [[["Q:", " choice 0-2"]], [["Q:", " choice 1-0"]]])
        self.assertEqual([q["pairs"][0][1] for q in tasks["words:bpb"]], [" a", " b", " c"])
        self.assertTrue(all(q["label"] == 0 and "uncond" not in q for q in tasks["quiz:bpb"]))

    def test_keeps_every_choice_for_accuracy_and_the_choices_without_the_question_for_acc_uncond(self):
        _, (_, _, tasks) = self.selected(["rc"])
        self.assertEqual([len(q["pairs"]) for q in tasks["quiz:rc"]], [3] * 6)
        self.assertNotIn("uncond", tasks["quiz:rc"][0])
        self.assertEqual(tasks["pick:rc"][0]["uncond"], [["Answer:", " choice 0-0"], ["Answer:", " choice 0-1"]])
        self.assertEqual([q["label"] for q in tasks["pick:rc"]], [1, 0])
        self.assertEqual(tasks["last"], [{"doc_id": 0, "label": 0, "idx": 0, "pairs": [["Q:", " end"]]}])

    def test_draws_the_same_questions_for_tasks_that_ask_the_same(self):
        _, (_, _, tasks) = self.selected(["bpb", "rc"], per_task=3, seed=4)
        self.assertEqual(len(tasks["quiz:bpb"]), 3)
        self.assertEqual([q["doc_id"] for q in tasks["quiz:bpb"]], [q["doc_id"] for q in tasks["quiz:rc"]])
        self.assertEqual(len(tasks["words:bpb"]), 3)

    def test_names_equal_selections_equally_and_others_differently(self):
        first, _ = self.selected(["bpb"])
        self.assertEqual(self.selected(["bpb"])[0], first)
        self.assertNotEqual(self.selected(["bpb"], per_task=2)[0], first)

    def test_refuses_a_metric_that_it_does_not_compute(self):
        tasks = {"odd": ({"task_name": "odd", "primary_metric": "f1"}, [request(0, 0, 0, " a")])}
        requests = make_requests(self.directory, tasks, {"suite": ["odd"]})
        with self.assertRaises(ValueError):
            om.select(requests, self.directory / "selections", ["suite"])


def characters(texts):
    return [[ord(c) for c in text] for text in texts]


class RowsTest(unittest.TestCase):
    def test_moves_the_contexts_trailing_spaces_to_the_continuation_after_the_bos_token(self):
        [(tokens, answer)] = om.encode_rows(characters, 1, [("The cat  ", "sat.")])
        self.assertEqual(tokens, [1, *characters(["The cat  sat."])[0]])
        self.assertEqual(answer, len("  sat."))

    def test_keeps_one_token_more_than_the_length_from_the_end_of_a_row(self):
        [(tokens, answer)] = om.encode_rows(characters, 1, [("abcdefgh", "ij")], length=4)
        self.assertEqual(tokens, characters(["fghij"])[0])
        self.assertEqual(answer, 2)

    def test_refuses_a_continuation_longer_than_the_length(self):
        with self.assertRaises(ValueError):
            om.encode_rows(characters, 1, [("a", "bcdef")], length=3)

    @unittest.skipUnless(importlib.util.find_spec("tokenizers") and importlib.util.find_spec("numpy"), "needs tokenizers")
    def test_keeps_a_selections_rows_for_a_tokenizer_and_gives_them_back_for_it_alone(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            tokenizer.train_from_iterator(["the cat sat on the mat"] * 10, trainers.BpeTrainer(vocab_size=300))
            tokenizer.save(str(directory / "tokenizer.json"))
            requests = make_requests(directory, {"t": ({"primary_metric": "acc_raw"}, multiple_choice([1, 0]))},
                                     {"s": ["t"]})
            selection = om.select(requests, directory, ["s"])
            made = om.rows(selection, directory, directory / "tokenizer.json", 0)
            expected = om.encode_rows(om.tokenizer_encode(directory / "tokenizer.json"), 0,
                                      om.pairs(om.read_selection(selection)[2]))
            kept = om.read_rows(made, selection, directory / "tokenizer.json")
            self.assertEqual([(list(row), answer) for row, answer in kept], expected)
            other = directory / "other.json"
            other.write_text((directory / "tokenizer.json").read_text() + "\n")
            with self.assertRaises(ValueError):
                om.read_rows(made, selection, other)


class WindowTest(unittest.TestCase):
    def test_reads_a_row_within_the_context_whole(self):
        self.assertEqual(om.windows(6, 2, 8), [(0, 6, 4)])

    def test_reads_a_longer_row_in_windows_that_end_at_its_end_and_step_back_by_half(self):
        self.assertEqual(om.windows(20, 12, 8), [(12, 20, 16), (8, 16, 12), (4, 12, 8)])

    def test_reads_only_the_windows_that_reach_the_answer(self):
        self.assertEqual(om.windows(20, 2, 8), [(12, 20, 18)])


@unittest.skipUnless(HAS_TORCH, "needs torch")
class ReaderTest(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.table = torch.randn(50, 50)

    def logits(self, inputs):
        # Each position's logits depend on its own token and the one before it.
        previous = inputs.roll(1, dims=1)
        previous[:, 0] = 0
        return self.table[inputs] + 0.5 * self.table[previous]

    def direct(self, tokens, answer):
        import torch

        tokens = torch.tensor(tokens)
        logits = self.logits(tokens[None, :-1])[0].log_softmax(-1)
        positions = range(len(tokens) - 1 - answer, len(tokens) - 1)
        picked = [logits[k, tokens[k + 1]].item() for k in positions]
        greedy = all(logits[k].argmax().item() == tokens[k + 1].item() for k in positions)
        return sum(picked), greedy

    def test_gives_each_rows_log_likelihood_and_greedy_flag_whatever_shares_its_batch(self):
        rng = random.Random(0)
        rows = [([rng.randrange(50) for _ in range(rng.randint(3, 7))], 0) for _ in range(9)]
        rows = [(tokens, rng.randint(1, len(tokens) - 1)) for tokens, _ in rows]
        expected = [self.direct(tokens, answer) for tokens, answer in rows]
        for batch_tokens, most_rows in ((1, None), (16, 2), (1000, None)):
            sums, greedy = om.reader(self.logits, 16, 0, "cpu", batch_tokens, most_rows)(rows)
            for (total, flag), (want, want_flag) in zip(zip(sums, greedy), expected):
                self.assertAlmostEqual(total, want, places=5)
                self.assertEqual(flag, want_flag)

    def test_reads_a_row_longer_than_the_context_from_the_windows_that_end_after_each_position(self):
        tokens = list(range(1, 21))
        sums, _ = om.reader(self.logits, 8, 0, "cpu")([(tokens, 12)])
        # The model sees one token back, so each window gives what the whole row gives.
        self.assertAlmostEqual(sums[0], self.direct(tokens, 12)[0], places=5)


def question(pairs, label=0, uncond=None):
    found = {"doc_id": 0, "label": label, "pairs": [["Q:", p] for p in pairs]}
    if uncond is not None:
        found["uncond"] = [["A:", p] for p in pairs]
    return found


class ScoreTest(unittest.TestCase):
    def test_scores_bits_per_byte_as_the_mean_of_the_answers_bits(self):
        questions = [question([" ab"]), question([" é"])]
        value = om.task_score("bits_per_byte_corr", questions, [-3.0, -1.5], [False, False], [1, 1])
        self.assertAlmostEqual(value, (om.LOG_2_OF_E * 3 / 3 + om.LOG_2_OF_E * 1.5 / 3) / 2)

    def test_gives_equal_choices_to_the_first(self):
        self.assertEqual(om.task_score("acc_raw", [question([" a", " b"], label=0)], [-1.0, -1.0], [0, 0], [1, 1]), 1)
        self.assertEqual(om.task_score("acc_raw", [question([" a", " b"], label=1)], [-1.0, -1.0], [0, 0], [1, 1]), 0)

    def test_scores_each_accuracy_by_its_own_measure(self):
        # The longer second choice has the lower sum but the higher sum per token, per character and per byte.
        pairs, totals, tokens = [" a", " bbbb"], [-2.0, -3.0], [1, 4]
        for metric, right in (("acc_raw", 0), ("acc_per_token", 1), ("acc_per_char", 1), ("acc_per_byte", 1)):
            for label in (0, 1):
                value = om.task_score(metric, [question(pairs, label)], totals, [0, 0], tokens)
                self.assertEqual(value, float(label == right), metric)
        uncond = question(pairs, label=1, uncond=True)
        self.assertEqual(om.task_score("acc_uncond", [uncond], totals + [-1.0, -5.0], [0] * 4, tokens * 2), 1)

    def test_scores_greedy_accuracy_by_whether_every_token_of_the_answer_is_the_likeliest(self):
        questions = [question([" a"]), question([" b"])]
        self.assertEqual(om.task_score("greedy_acc", questions, [-1.0, -2.0], [True, False], [1, 1]), 0.5)

    def test_averages_each_group_as_one_member_of_its_suite(self):
        groups = {"suite": ["group", "c"], "group": ["a", "b"]}
        self.assertEqual(om.suite_scores(groups, {"a": 1.0, "b": 3.0, "c": 4.0}), {"suite": 3.0})


if __name__ == "__main__":
    unittest.main()
