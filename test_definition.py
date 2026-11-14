import tempfile
import unittest
from pathlib import Path

import olmes_measure as om
from language_model_evaluation import definition, overlap, paloma
from test_olmes_measure import make_requests, request
from test_paloma import CATEGORIES, fetched, members_files

TOY = {"name": "toy", "compare": "mean_rate", "olmes": {"suites": ["first", "second"]},
       "categories": {"science": {"olmes": ["t1", {"hard": ["t2", "t3"]}]}, "sums": {"olmes": ["second"]},
                      "web": {"paloma": CATEGORIES["web"]}, "books": {"paloma": CATEGORIES["books"]}}}
GROUPS = {"first": ["t1", "pair"], "pair": ["t2", "t3"], "second": ["t4", "inner"], "inner": ["t5", "t6"]}
# Two languages: English as TOY's, Japanese questions and text beside them.
BOTH = {**TOY, "compare": "mean_rate_by_language", "japanese": {"suites": ["ja"]},
        "categories": {**TOY["categories"], "ja_common": {"japanese": ["j1", "j2"]},
                       "ja_web": {"japanese_text": ["mc4_ja"]}},
        "languages": {"en": ["science", "sums", "web", "books"], "ja": ["ja_common", "ja_web"]}}


class LoadTest(unittest.TestCase):
    def test_a_definition_within_another_says_what_differs_and_takes_the_rest(self):
        whole, part = definition.load("knowledge-en-1"), definition.load("knowledge-en-part-1")
        self.assertNotIn("within", whole)
        self.assertEqual((part["name"], part["within"]), ("knowledge-en-part-1", "knowledge-en-1"))
        for key in ("categories", "compare", "overlap"):
            self.assertEqual(part[key], whole[key])
        self.assertEqual(part["olmes"]["requests"], whole["olmes"]["requests"])
        self.assertEqual(part["olmes"]["suites"], whole["olmes"]["suites"])  # what the categories count
        self.assertEqual(part["olmes"]["per_task"], 100)
        self.assertNotEqual(part["olmes"]["selection"], whole["olmes"]["selection"])
        self.assertEqual(part["paloma"]["whole"], whole["paloma"]["text"])  # the text that the part is cut from
        self.assertNotEqual(part["paloma"]["text"], whole["paloma"]["text"])
        self.assertEqual(len(definition.categories(part)), 11)
        self.assertEqual(list(definition.members(whole)), ["web", "encyclopedia", "papers", "posts", "books", "news"])


class TwoLanguagesTest(unittest.TestCase):
    def test_knowledge_ja_en_measures_knowledge_en_beside_japanese_and_counts_the_two_languages_equally(self):
        english, both = definition.load("knowledge-en-1"), definition.load("knowledge-ja-en-1")
        for section in ("olmes", "paloma"):
            self.assertEqual(both[section], english[section])
        self.assertEqual({k: v for k, v in both["overlap"].items() if k != "japanese"}, english["overlap"])
        self.assertEqual(list(both["categories"])[:11], list(english["categories"]))
        self.assertEqual({language: len(names) for language, names in both["languages"].items()}, {"en": 11, "ja": 6})
        self.assertEqual(both["compare"], "mean_rate_by_language")
        self.assertEqual(list(definition.members(both, "japanese_text")), ["ja_web", "ja_encyclopedia"])


class ScoresTest(unittest.TestCase):
    def test_a_category_that_is_answered_is_the_mean_of_its_tasks_groups_and_named_means(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0}
        found = definition.answered(TOY, GROUPS, scores)
        self.assertEqual(found, {"science": (1.0 + 3.0) / 2, "sums": (6.0 + 2.0) / 2})  # a group counts as one member
        self.assertEqual(found["sums"], om.suite_scores(GROUPS, scores)["second"])

    def test_the_categories_count_every_task_of_the_suites_once_unless_the_selection_is_partial(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0}
        twice = dict(TOY, categories={**TOY["categories"], "again": {"olmes": ["t1"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(twice, GROUPS, scores)
        fewer = dict(TOY, categories={"science": {"olmes": ["t1"]}, "sums": {"olmes": ["second"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(fewer, GROUPS, scores)  # t2 and t3 are asked and not counted
        # a reserve that lacks t3 and all of the inner group: its groups hold what is left
        left = {"t1": 1.0, "t2": 2.0, "t4": 6.0}
        pruned = {"first": ["t1", "pair"], "pair": ["t2"], "second": ["t4"]}
        with self.assertRaisesRegex(ValueError, "does not hold"):
            definition.answered(TOY, pruned, left)
        self.assertEqual(definition.answered(TOY, pruned, left, partial=True), {"science": 1.5, "sums": 6.0})

    def test_a_category_that_is_read_is_the_mean_of_its_members_and_a_source_the_mean_of_its_domains(self):
        domains = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 2.0, "c4_100_domains/2_b": 4.0,
                   "dolma-v1_5/common-crawl": 5.0, "dolma-v1_5/books": 7.0, "dolma-v1_5/wiki": 100.0}
        found = definition.read(TOY, domains)
        self.assertEqual(found, {"web": (1.0 + 3.0 + 5.0) / 3, "books": 7.0})  # the wiki is no member of these
        del domains["dolma-v1_5/common-crawl"]
        with self.assertRaisesRegex(ValueError, "common-crawl"):
            definition.read(TOY, domains)
        self.assertEqual(definition.read(TOY, domains, partial=True)["web"], 2.0)
        del domains["dolma-v1_5/books"]
        with self.assertRaisesRegex(ValueError, "books"):
            definition.read(TOY, domains, partial=True)  # a category with no member at all has no score

    def test_answers_and_reads_the_japanese_categories_beside_the_english_ones(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0, "j1": 1.0, "j2": 2.0}
        groups = {**GROUPS, "ja": ["j1", "j2"]}
        found = definition.answered(BOTH, groups, scores)
        self.assertEqual(found, {"science": 2.0, "sums": 4.0, "ja_common": 1.5})
        fewer = dict(BOTH, categories={**BOTH["categories"], "ja_common": {"japanese": ["j1"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(fewer, groups, scores)  # j2 is asked and not counted
        domains = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 3.0, "dolma-v1_5/common-crawl": 5.0,
                   "dolma-v1_5/books": 7.0, "mc4_ja/mc4_ja": 0.9}
        self.assertEqual(definition.read(BOTH, domains), {"web": 3.0, "books": 7.0, "ja_web": 0.9})

    def test_each_language_counts_equally_however_many_categories_it_has(self):
        standing = {category: 2.0 for category in definition.categories(BOTH)}
        better_english = dict(standing, science=1.0)  # one of four English categories half the bits
        better_japanese = dict(standing, ja_web=1.0)  # one of two Japanese categories half the bits
        self.assertAlmostEqual(definition.improvement(BOTH, better_english, standing), (0.5 / 4) / 2)
        self.assertAlmostEqual(definition.improvement(BOTH, better_japanese, standing), (0.5 / 2) / 2)
        missing = dict(BOTH, languages={"en": ["science", "sums", "web", "books"], "ja": ["ja_common"]})
        with self.assertRaisesRegex(ValueError, "every category"):
            definition.improvement(missing, better_english, standing)

    def test_one_result_improves_on_another_by_the_mean_of_the_categories_rates(self):
        standing = {category: 2.0 for category in definition.categories(TOY)}
        candidate = dict(standing, web=1.0, sums=2.2)  # one category half the bits, one a tenth more
        self.assertEqual(definition.categories(TOY), ("science", "sums", "web", "books"))
        self.assertAlmostEqual(definition.improvement(TOY, candidate, standing), (0.5 - 0.1) / 4)
        self.assertEqual(definition.improvement(TOY, standing, standing), 0)
        with self.assertRaises(ValueError):
            definition.improvement(dict(TOY, compare="other"), candidate, standing)


class MakeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        asked = lambda count: [dict(request(doc_id, 0, 0, " yes", context=f"Is {doc_id} even?"),
                                    doc={"query": f"Question: is the number {doc_id} even, odd, or neither of them?"})
                               for doc_id in range(count)]
        tasks = {"t1": ({"task_name": "t1", "primary_metric": "bits_per_byte_corr"}, asked(5)),
                 "t2": ({"task_name": "t2", "primary_metric": "bits_per_byte_corr"}, asked(3))}
        self.requests = make_requests(self.directory, tasks, {"first": ["t1", "t2"]})
        self.files = {**members_files("val"), **members_files("test")}
        self.listed = lambda source, split: sorted(p for p in self.files if p.startswith(f"{source}/{split}/"))

    def download(self, path, staging):
        import gzip
        import json
        local = Path(staging) / path.replace("/", "-")
        with gzip.open(local, "wt", encoding="utf-8") as file:
            file.writelines(json.dumps({"text": text}) + "\n" for text in self.files[path])
        return local

    def named(self):
        """A definition of the toy data, with the names that the tools give its artifacts."""
        out = self.directory / "first"
        selection = om.select(self.requests, out, ["first"], per_task=2)
        aside = om.reserve(self.requests, selection, out, per_task=2)
        categories = {"asked": {"olmes": ["first"]}, **{name: {"paloma": parts} for name, parts in CATEGORIES.items()}}
        val = paloma.fetch(out, "val", CATEGORIES, self.download, self.listed)
        test = paloma.fetch(out, "test", CATEGORIES, self.download, self.listed)
        sizes = dict(member_bytes=100, domain_bytes=30, document_bytes=10)
        return {"name": "toy", "compare": "mean_rate", "categories": categories,
                "olmes": {"requests": self.requests.name, "suites": ["first"], "per_task": 2, "seed": 0,
                          "selection": selection.name, "reserve": {"per_task": 2, "seed": 1, "selection": aside.name}},
                "paloma": {"split": "val", "whole": val.name, "part": sizes,
                           "text": paloma.part(val, out, CATEGORIES, **sizes).name,
                           "reserve": {"split": "test", "whole": test.name,
                                       "text": paloma.part(test, out, CATEGORIES, **sizes).name}},
                "overlap": {"paloma": {"val": val.name, "test": test.name},
                            "artifact": overlap.make(out, self.requests, [val, test]).name}}

    def test_makes_every_artifact_that_a_definition_names_under_its_name(self):
        toy = self.named()
        made = definition.make(toy, self.directory / "second", self.requests, self.download, self.listed)
        self.assertEqual(len(made), 7)
        self.assertEqual(set(made), {toy["olmes"]["selection"], toy["olmes"]["reserve"]["selection"],
                                     toy["paloma"]["whole"], toy["paloma"]["text"], toy["paloma"]["reserve"]["whole"],
                                     toy["paloma"]["reserve"]["text"], toy["overlap"]["artifact"]})
        for name, artifact in made.items():
            self.assertEqual(om.check(artifact)["kind"], name.rsplit("-", 1)[0])
        # what is there already is checked and kept: nothing is fetched again
        refuse = lambda *arguments: self.fail("fetched again")
        self.assertEqual(definition.make(toy, self.directory / "second", self.requests, refuse, refuse), made)

    def test_refuses_what_comes_out_under_another_name_and_other_requests(self):
        toy = self.named()
        other = dict(toy, olmes=dict(toy["olmes"], selection="selection-0000000000000000"))
        with self.assertRaisesRegex(ValueError, "selection-0000000000000000"):
            definition.make(other, self.directory / "third", self.requests, self.download, self.listed)
        with self.assertRaisesRegex(ValueError, "is made from"):
            definition.make(dict(toy, olmes=dict(toy["olmes"], requests="requests-0")), self.directory / "third",
                            self.requests, self.download, self.listed)


if __name__ == "__main__":
    unittest.main()
