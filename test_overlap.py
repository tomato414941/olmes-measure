import gzip
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import olmes_measure as om
from language_model_evaluation import overlap, paloma
from test_paloma import fetched, members_files

QUESTION = "Which of the following best explains why the sky appears blue to an observer on the ground at noon?"
PARAGRAPH = "The quick brown fox jumps over the lazy dog and then it runs far away."


class QuestionTest(unittest.TestCase):
    def test_finds_a_question_by_the_middle_of_its_letters_and_digits_whatever_the_case_marks_and_space(self):
        key, letters = overlap.question_key(QUESTION), overlap.normalized(QUESTION)
        self.assertTrue(letters.startswith("whichofthefollowingbest"))
        self.assertEqual(len(key), overlap.WIDTH)
        self.assertEqual(letters.index(key), (len(letters) - overlap.WIDTH) // 2)
        # what OLMES wraps a question in is not part of it; one too short to tell from other text has no key
        self.assertEqual(overlap.question_key(f"Question: {QUESTION}\nAnswer:"), key)
        self.assertEqual(overlap.question_key("The following are questions (with answers) about physics.\n\n" + QUESTION),
                         key)
        self.assertIsNone(overlap.question_key("What is 2 + 2?"))
        self.assertEqual(overlap.keys_of(["What is 2 + 2?", QUESTION]), [key])
        found = overlap.Overlap(overlap.keys_of([QUESTION]), b"")
        self.assertTrue(found.holds_question("A page.\n\nWHICH of the following -- best explains why the sky\nappears "
                                             "blue to an observer on the ground, at noon? Scattering."))
        self.assertFalse(found.holds_question("Which of the following best explains why the sea is salty?"))
        self.assertFalse(overlap.Overlap([], b"").holds("anything at all"))

    def test_keeps_a_texts_letters_and_digits_the_same_with_and_without_characters_beyond_ascii(self):
        plain = "Fig. 3_a (2019):\tThe END -- of it_all\n"
        self.assertEqual(overlap.normalized(plain), "fig3a2019theendofitall")
        self.assertEqual(overlap.normalized(plain + "é"), "fig3a2019theendofitallé")  # the same, by Unicode's rules
        self.assertEqual(overlap.normalized("Ｃａｆé ２０１９ — ǅ"), "café2019dž")  # compatible forms, case folded


class TextTest(unittest.TestCase):
    def test_finds_a_paragraph_when_a_line_of_the_document_is_one_of_the_long_lines(self):
        found = overlap.Overlap([], overlap.digests_of(paloma.paragraphs(f"A short line.\n  {PARAGRAPH}\n" * 2)))
        self.assertEqual(len(found.first), 1)  # the long line, once
        self.assertTrue(found.holds_paragraph(f"Something else.\n\t{PARAGRAPH}  \nAnd more."))
        self.assertTrue(found.holds(PARAGRAPH))
        self.assertFalse(found.holds(f"Something else. {PARAGRAPH} And more, all on one line.\nA short line."))
        many = overlap.Overlap([], overlap.digests_of(f"{PARAGRAPH} {number}" for number in range(100)))
        self.assertTrue(all(many.holds(f"{PARAGRAPH} {number}") for number in (0, 37, 99)))
        self.assertFalse(many.holds(f"{PARAGRAPH} 100"))

    def test_finds_a_stretch_of_the_text_however_it_is_spaced_cased_marked_or_broken_into_lines(self):
        article = " ".join(f"The {number} lobsters of the northern coast are caught in pots , weighed and sold ."
                           for number in range(40))  # as Paloma's WikiText stands: spaced around its marks
        found = overlap.Overlap([], b"", overlap.stretches_of([article, "too short to hold a stretch"]))
        letters = overlap.normalized(article)
        self.assertEqual(len(found.hashes), len(range(0, len(letters) - overlap.LETTERS + 1, overlap.STRIDE)))
        # the same text as an encyclopedia writes it, in other lines: no line is the same, and the letters are
        copy = article.replace(" ,", ",").replace(" .", ".").upper().replace("POTS, ", "POTS,\n")
        self.assertFalse(set(copy.split("\n")) & set(article.split("\n")))
        self.assertTrue(found.holds_stretch(copy) and found.holds(copy))
        # any LETTERS + STRIDE letters and digits in a row that the two share hold a stretch that is looked for
        share = overlap.LETTERS + overlap.STRIDE
        for start in (0, 1, 77, 500, len(letters) - share):
            self.assertTrue(found.holds_stretch("", "x" * 50 + letters[start:start + share] + "y" * 50))
        self.assertFalse(found.holds_stretch("", letters[1:overlap.LETTERS]))  # fewer letters than a stretch
        self.assertFalse(found.holds("Crabs of the southern coast are another matter, and nobody weighs them. " * 20))
        # every stretch's hash, from one running sum: the sum of code[i + k] x MULTIPLIER^k
        by_hand = [sum(ord(c) * pow(overlap.MULTIPLIER, k, 1 << 64) for k, c in enumerate(letters[i:i + 5])) % (1 << 64)
                   for i in range(len(letters[:40]) - 4)]
        self.assertEqual(overlap.stretches(letters[:40], 5).tolist(), by_hand)

    def test_tells_the_same_of_many_documents_among_several_processes(self):
        found = overlap.Overlap(overlap.keys_of([QUESTION]), overlap.digests_of([PARAGRAPH]))
        texts = ["one", f"copied: {QUESTION}", "two", f"quoted\n{PARAGRAPH}", "three"] * 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # the test's own threads, which no process uses
            self.assertEqual(found.among(2)(texts), [found.holds(text) for text in texts])
        self.assertEqual(found.among(2)(texts[:5]), [False, True, False, True, False])


class ArtifactTest(unittest.TestCase):
    def test_tells_the_questions_of_the_requests_and_the_text_of_palomas_artifacts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        directory = Path(directory.name)
        work = directory / "work"
        (work / "requests").mkdir(parents=True)
        documents = [{"query": f"Question: {QUESTION}\nAnswer:", "choices": ["a", "b"]}, {"question": "What is 2 + 2?"}]
        with gzip.open(work / "requests" / "task.jsonl.gz", "wt", encoding="utf-8") as file:
            file.writelines(json.dumps({"doc": document}) + "\n" for document in documents)
        requests = om.seal(work, "requests", olmes="test")
        texts = members_files()
        texts["ptb/val/val.jsonl.gz"] = [f"A short line.\n{PARAGRAPH}"]
        text = fetched(directory, texts)
        artifact = overlap.make(directory, requests, [text])
        manifest = om.check(artifact)
        self.assertTrue(artifact.name.startswith("overlap-"))
        self.assertEqual((manifest["questions"], manifest["text"]), (requests.name, [text.name]))
        self.assertEqual((artifact / "questions.txt").read_text(), overlap.question_key(QUESTION) + "\n")
        self.assertEqual((artifact / "paragraphs.bin").read_bytes(), paloma.digest(PARAGRAPH))
        self.assertEqual((artifact / "stretches.bin").read_bytes(),
                         overlap.stretches_of(text for texts_ in texts.values() for text in texts_))
        found = overlap.load(artifact)
        self.assertTrue(found.holds(f"Homework help: {QUESTION.lower()}"))
        self.assertTrue(found.holds(f"Old news.\n{PARAGRAPH}"))
        self.assertFalse(found.holds("What is 2 + 2?\nA short line."))
        with self.assertRaisesRegex(ValueError, "no requests artifact"):
            overlap.make(directory, artifact, [])


if __name__ == "__main__":
    unittest.main()
