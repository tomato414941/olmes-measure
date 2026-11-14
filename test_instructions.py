import json
import tempfile
import unittest
from pathlib import Path

from language_model_evaluation import definition, instructions


def prompt(key, text):
    return {"key": key, "prompt": text, "instruction_id_list": ["ja:length_constraints:number_sentences"],
            "kwargs": [{"relation": "at least", "num_sentences": 3}]}


class PromptsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        (self.home / "data").mkdir()
        rows = [prompt(1, "3文以上の文章で、雨について書いてください。"), prompt(2, "雪について書いてください。")]
        (self.home / "data" / "ja_input_data.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_gives_a_languages_prompts_as_the_authors_give_them(self):
        self.assertEqual([row["key"] for row in instructions.prompts("ja", self.home)], [1, 2])
        with self.assertRaises(ValueError):
            instructions.prompts("de", self.home)

    def test_scores_only_replies_that_answer_each_prompt_once(self):
        asked = instructions.prompts("ja", self.home)
        with self.assertRaisesRegex(ValueError, "each of the prompts once"):
            instructions.evaluated(asked, {"雪について書いてください。": "雪。"}, self.home / "out", self.home, None)

    def test_asks_a_prompt_as_its_reply_was_asked_when_that_says_lines_for_sentences(self):
        asked = instructions.prompts("ja", self.home)
        replies = {"3行以上の文章で、雨について書いてください。": "雨。", "雪について書いてください。": "雪。"}
        self.assertEqual([row["prompt"] for row in instructions.as_replied(asked, replies)], list(replies))
        with self.assertRaisesRegex(ValueError, "no reply answers"):
            instructions.as_replied(asked, {"雪について書いてください。": "雪。", "雨とは別の話です。": "。"})


class FollowedTest(unittest.TestCase):
    def test_a_higher_share_of_prompts_followed_is_the_better_and_each_language_counts_equally(self):
        chosen = definition.load("instructions-ja-en-1")
        self.assertEqual(chosen["instructions"]["commit"], instructions.COMMIT)
        standing = definition.followed(chosen, {"en": 0.30, "ja": 0.15})
        self.assertEqual(standing, {"ifeval": 0.30, "ja_ifeval": 0.15})
        better = dict(standing, ja_ifeval=0.18)
        self.assertAlmostEqual(definition.improvement(chosen, better, standing), (0.18 / 0.15 - 1) / 2)
        self.assertLess(definition.improvement(chosen, standing, better), 0)


if __name__ == "__main__":
    unittest.main()
