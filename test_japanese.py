import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import olmes_measure as om
from language_model_evaluation import japanese


def jcqa_row(n, label=1):
    return {"q_id": n, "question": f"問{n}は？", **{f"choice{k}": f"答{n}-{k}" for k in range(5)}, "label": label}


def kuci_row(n, label="c"):
    return {"id": n, "context": f"文 {n} を 書いたら", **{f"choice_{k}": f"続き {n} {k}" for k in "abcd"},
            "label": label}


def jsquad(questions):
    return {"data": [{"title": "題", "paragraphs": [{"context": f"題 [SEP] 本文{n}。", "qas": [
        {"id": f"q{n}", "question": f"問{n}は？", "answers": [{"text": f"答{n}", "answer_start": 0}]}]}
        for n in range(questions)]}]}


def jmmlu_zip(subjects):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for subject, rows in subjects.items():
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(["question", "A", "B", "C", "D", "answer"])
            writer.writerows(rows)
            archive.writestr(f"JMMLU/test/{subject}.csv", "﻿" + text.getvalue())
        archive.writestr("__MACOSX/JMMLU/test/._history.csv", "")
    return data.getvalue()


class JapaneseTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        lines = lambda rows: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self.files = {
            ("jglue", "datasets/jcommonsenseqa-v1.2/train-v1.2.json"): lines(jcqa_row(n, 0) for n in range(100, 106)),
            ("jglue", "datasets/jcommonsenseqa-v1.2/valid-v1.2.json"): lines([jcqa_row(1), jcqa_row(2, 4)]),
            ("kuci", "dataset/train.jsonl"): lines(kuci_row(n, "a") for n in range(100, 106)),
            ("kuci", "dataset/test.jsonl"): lines([kuci_row(1)]),
            ("xwinograd", "test/xwinograd-ja.json"): lines([{"sentence1": "犬は猫を追った。犬は速かった。",
                                                              "sentence2": "犬は猫を追った。猫は速かった。",
                                                              "answer": "2"}]),
            ("jglue", "datasets/jsquad-v1.2/train-v1.2.json"): json.dumps(jsquad(3), ensure_ascii=False),
            ("jglue", "datasets/jsquad-v1.2/valid-v1.2.json"): json.dumps(jsquad(1), ensure_ascii=False),
            ("jmmlu", "JMMLU.zip"): jmmlu_zip({"history": [["年は？", "一", "二", "三", "四", "C"]],
                                               "physics": [["力は？", "甲", "乙", "丙", "丁", "A"]]}),
        }
        patcher = mock.patch.object(japanese, "fetch", self.fetch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, source, path, raw):
        local = self.directory / source / path
        local.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[(source, path)]
        local.write_bytes(data if isinstance(data, bytes) else data.encode())
        return local

    def test_asks_a_jcommonsenseqa_question_after_five_training_questions_with_their_answers(self):
        contexts, continuations, label = japanese.jcommonsenseqa(self.directory)[0]
        self.assertEqual(continuations, [f"答1-{k}" for k in range(5)])
        self.assertEqual(label, 1)
        self.assertEqual(len(set(contexts)), 1)  # every choice after the same question
        context = contexts[0]
        self.assertTrue(context.endswith("\n\n質問：問1は？\n回答："))
        examples = context.split("\n\n")[:-1]
        self.assertEqual(len(examples), 5)
        for example in examples:
            question = example.removeprefix("質問：").partition("\n")[0]
            n = int(question.removeprefix("問").removesuffix("は？"))
            self.assertIn(n, range(100, 106))  # from the training split, with its answer
            self.assertEqual(example, f"質問：問{n}は？\n回答：答{n}-0")
        # the same examples before every question
        self.assertEqual(japanese.jcommonsenseqa(self.directory)[1][0][0].rpartition("質問：")[0],
                         context.rpartition("質問：")[0])

    def test_reads_the_kuci_clause_that_follows_with_the_words_joined_as_japanese_writes_them(self):
        contexts, continuations, label = japanese.kuci(self.directory)[0]
        self.assertEqual(continuations, ["続き1a", "続き1b", "続き1c", "続き1d"])
        self.assertEqual(label, 2)
        self.assertTrue(contexts[0].endswith("\n\n文1を書いたら"))
        self.assertEqual(contexts[0].count("\n\n"), 5)

    def test_reads_the_rest_of_an_xwinograd_sentence_after_each_referent(self):
        contexts, continuations, label = japanese.xwinograd(self.directory)[0]
        self.assertEqual(contexts, ["犬は猫を追った。犬", "犬は猫を追った。猫"])
        self.assertEqual(continuations, ["は速かった。", "は速かった。"])
        self.assertEqual(label, 1)

    def test_asks_a_jsquad_question_of_its_passage_without_the_articles_title(self):
        contexts, continuations, label = japanese.jsquad(self.directory)[0]
        self.assertEqual((continuations, label), (["答0"], 0))
        self.assertTrue(contexts[0].endswith("\n\n文脈：本文0。\n質問：問0は？\n回答："))
        self.assertEqual(contexts[0].count("文脈："), 3)  # two examples of the training split, then the question
        self.assertNotIn("[SEP]", contexts[0])

    def test_asks_each_jmmlu_subject_by_itself_without_examples(self):
        subjects = japanese.jmmlu(self.directory)
        self.assertEqual(sorted(subjects), ["history", "physics"])
        self.assertEqual(subjects["history"], [(["質問：年は？\n回答："] * 4, ["一", "二", "三", "四"], 2)])

    def test_makes_an_artifact_whose_selection_reads_each_questions_correct_answer(self):
        artifact = japanese.make(self.directory / "out", self.directory)
        suites = json.loads((artifact / "suites.json").read_text())
        self.assertEqual(suites["ja:bpb"], ["jcommonsenseqa:rc:bpb::ja", "jsquad:rc:bpb::ja", "kuci:rc:bpb::ja",
                                            "xwinograd_ja:rc:bpb::ja", "jmmlu:rc:bpb::ja"])
        self.assertEqual(suites["jmmlu:rc:bpb::ja"], ["jmmlu_history:rc:bpb::ja", "jmmlu_physics:rc:bpb::ja"])
        selection = om.select(artifact, self.directory / "out", ["ja:bpb"])
        _, metrics, tasks = om.read_selection(selection)
        self.assertEqual(set(metrics.values()), {"bits_per_byte_corr"})
        answers = {alias: [question["pairs"][0][1] for question in asked] for alias, asked in tasks.items()}
        self.assertEqual(answers, {"jcommonsenseqa:rc:bpb::ja": ["答1-1", "答2-4"], "jsquad:rc:bpb::ja": ["答0"],
                                   "kuci:rc:bpb::ja": ["続き1c"], "xwinograd_ja:rc:bpb::ja": ["は速かった。"],
                                   "jmmlu_history:rc:bpb::ja": ["三"], "jmmlu_physics:rc:bpb::ja": ["甲"]})
        self.assertEqual(tasks["xwinograd_ja:rc:bpb::ja"][0]["pairs"][0][0], "犬は猫を追った。猫")


class FetchTest(unittest.TestCase):
    def test_refuses_a_kept_file_that_is_not_the_one_at_the_sources_revision(self):
        with tempfile.TemporaryDirectory() as raw:
            local = Path(raw, "kuci", "dataset", "test.jsonl")
            local.parent.mkdir(parents=True)
            local.write_text("{}\n")
            with self.assertRaises(ValueError):
                japanese.fetch("kuci", "dataset/test.jsonl", raw)


if __name__ == "__main__":
    unittest.main()
