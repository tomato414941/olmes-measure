import csv
import gzip
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import olmes_measure as om
from language_model_evaluation import japanese, japanese_text, overlap, paloma
from test_olmes_measure import make_requests, request


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

    def fetch(self, source, path, raw, sources=None):
        local = self.directory / source / path
        local.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[(source, path)]
        local.write_bytes(data if isinstance(data, bytes) else data.encode())
        return local

    def test_asks_a_jcommonsenseqa_question_after_five_training_questions_with_their_answers(self):
        contexts, continuations, label, question = japanese.jcommonsenseqa(self.directory)[0]
        self.assertEqual(continuations, [f"答1-{k}" for k in range(5)])
        self.assertEqual((label, question), (1, "問1は？"))
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
        contexts, continuations, label, question = japanese.kuci(self.directory)[0]
        self.assertEqual(continuations, ["続き1a", "続き1b", "続き1c", "続き1d"])
        self.assertEqual((label, question), (2, "文1を書いたら続き1c"))
        self.assertTrue(contexts[0].endswith("\n\n文1を書いたら"))
        self.assertEqual(contexts[0].count("\n\n"), 5)

    def test_reads_the_rest_of_an_xwinograd_sentence_after_each_referent(self):
        contexts, continuations, label, question = japanese.xwinograd(self.directory)[0]
        self.assertEqual(contexts, ["犬は猫を追った。犬", "犬は猫を追った。猫"])
        self.assertEqual(continuations, ["は速かった。", "は速かった。"])
        self.assertEqual((label, question), (1, "犬は猫を追った。猫は速かった。"))

    def test_asks_a_jsquad_question_of_its_passage_without_the_articles_title(self):
        contexts, continuations, label, question = japanese.jsquad(self.directory)[0]
        self.assertEqual((continuations, label, question), (["答0"], 0, "問0は？"))
        self.assertTrue(contexts[0].endswith("\n\n文脈：本文0。\n質問：問0は？\n回答："))
        self.assertEqual(contexts[0].count("文脈："), 3)  # two examples of the training split, then the question
        self.assertNotIn("[SEP]", contexts[0])

    def test_asks_each_jmmlu_subject_by_itself_without_examples(self):
        subjects = japanese.jmmlu(self.directory)
        self.assertEqual(sorted(subjects), ["history", "physics"])
        self.assertEqual(subjects["history"], [(["質問：年は？\n回答："] * 4, ["一", "二", "三", "四"], 2, "年は？")])

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

    def test_makes_the_same_artifact_from_the_same_files(self):
        first = japanese.make(self.directory / "first", self.directory)
        self.assertEqual(japanese.make(self.directory / "second", self.directory).name, first.name)

    def test_says_that_its_questions_are_japanese_and_gives_each_question_as_the_task_writes_it(self):
        artifact = japanese.make(self.directory / "out", self.directory)
        self.assertEqual(om.check(artifact)["language"], "ja")
        with gzip.open(artifact / "requests" / "jsquad:rc:bpb::ja.jsonl.gz", "rt", encoding="utf-8") as lines:
            self.assertEqual([json.loads(line)["doc"] for line in lines], [{"question": "問0は？"}])


LONG = ("気象庁によると、この夏は太平洋高気圧の張り出しが強く、各地で観測史上最も高い気温を記録した。"
        "とくに内陸の盆地では夜になっても気温が下がらず、熱中症で運ばれる人が相次いだ。")


class TextTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.web = {"a.json.gz": ["一つ目の文書。", LONG], "b.json.gz": ["三つ目の文書。"], "c.json.gz": ["取っておく文書。"]}
        splits = {"val": {"mc4_ja": ["a.json.gz", "b.json.gz"]}, "test": {"mc4_ja": ["c.json.gz"]}}
        for patcher in (mock.patch.object(japanese_text, "SPLITS", splits),
                        mock.patch.object(japanese_text, "fetch_file", self.fetch)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fetch(self, source, path, raw, sources):
        local = self.directory / "raw" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(local, "wt", encoding="utf-8") as lines:  # as mC4 writes a document
            lines.writelines(json.dumps({"text": text, "timestamp": "2019", "url": "https://example.com"},
                                        ensure_ascii=False) + "\n" for text in self.web[path])
        return local

    def test_writes_a_wiki40b_article_as_the_lines_of_a_text(self):
        written = "\n_START_ARTICLE_\n猛暑\n_START_SECTION_\n原因\n_START_PARAGRAPH_\n暑かった。_NEWLINE_とても。\n"
        self.assertEqual(japanese_text.article(repr(written.encode("utf-8"))), "猛暑\n原因\n暑かった。\nとても。")

    def test_keeps_a_split_of_the_japanese_text_as_paloma_keeps_its_text(self):
        artifact = japanese_text.fetch(self.directory / "out", "val", self.directory / "raw")
        manifest = om.check(artifact)
        self.assertEqual((manifest["kind"], manifest["language"]), ("japanese-text-val", "ja"))
        self.assertTrue(japanese_text.made_here(manifest))
        found = paloma.files(artifact)
        self.assertEqual(list(found), [("mc4_ja", "mc4_ja")])  # the source is one domain
        self.assertEqual(list(paloma.documents(found["mc4_ja", "mc4_ja"])), ["一つ目の文書。", LONG, "三つ目の文書。"])
        part = paloma.part(artifact, self.directory / "out", {"ja_web": ["mc4_ja"]}, member_bytes=60,
                           domain_bytes=60, document_bytes=30)
        self.assertEqual(list(paloma.documents(paloma.files(part)["mc4_ja", "mc4_ja"]))[0], "一つ目の文書。")
        # the same text, the same artifact
        self.assertEqual(japanese_text.fetch(self.directory / "again", "val", self.directory / "raw").name,
                         artifact.name)

    def test_finds_the_documents_that_hold_a_japanese_question_or_a_stretch_of_the_japanese_text(self):
        question = "次のうち、地上の観測者から見て正午に空が青く見える理由を最もよく説明しているものはどれですか。"
        work = Path(tempfile.mkdtemp(dir=self.directory))
        (work / "requests").mkdir()
        with gzip.open(work / "requests" / "q.jsonl.gz", "wt", encoding="utf-8") as lines:
            lines.write(json.dumps({**request(0, 0, 0, "答え"), "doc": {"question": question}}, ensure_ascii=False))
        (work / "tasks.json").write_text("{}")
        (work / "suites.json").write_text("{}")
        asked = om.seal(work, "requests", olmes=None, language="ja")
        text = japanese_text.fetch(self.directory / "out", "val", self.directory / "raw")
        artifact = overlap.make(self.directory / "out", [asked], [text])
        self.assertEqual(om.check(artifact)["widths"], {"ja": [32, 12]})
        found = overlap.load(artifact)
        self.assertTrue(found.holds("宿題です。" + question[6:] + "答えなさい。"))  # all but its first words
        self.assertTrue(found.holds("引用：" + LONG.replace("、", "，")))
        self.assertFalse(found.holds("今日は晴れて、公園で子どもたちが遊んでいた。"))

    def test_makes_an_artifact_of_olmess_questions_alone_as_before(self):
        english = make_requests(self.directory, {"t": ({}, [{**request(0, 0, 0, "x"), "doc": {
            "query": "Which of the following best explains why the sky appears blue at noon?"}}])}, {"s": ["t"]})
        self.assertNotIn("widths", om.check(overlap.make(self.directory / "out", english, [])))
        self.assertEqual(overlap.make(self.directory / "out", english, []),
                         overlap.make(self.directory / "out", [english], []))


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
