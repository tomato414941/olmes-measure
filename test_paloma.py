import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

import olmes_measure as om
from language_model_evaluation import paloma

CATEGORIES = {"web": ["c4_en", "c4_100_domains", "dolma-v1_5/common-crawl"], "books": ["dolma-v1_5/books"],
              "news": ["ptb"]}


def members_files(split="val", categories=CATEGORIES):
    """A file of each member of the categories, as Paloma names them: {path in the repository: its documents}."""
    found = {}
    for source, name in paloma.members(categories):
        file = f"{split}_{name}.jsonl.gz" if name else f"{split}.jsonl.gz"
        found[f"{source}/{split}/{file}"] = [f"the text of {source} {name}"]
    return found


def fetched(directory, texts, split="val", categories=CATEGORIES):
    """The artifact that `fetch` makes of a repository that holds these files."""

    def download(path, staging):
        local = Path(staging) / path.replace("/", "-")
        with gzip.open(local, "wt", encoding="utf-8") as file:
            file.writelines(json.dumps({"text": text}) + "\n" for text in texts[path])
        return local

    listed = lambda source, split: sorted(path for path in texts if path.startswith(f"{source}/{split}/"))
    return paloma.fetch(directory, split, categories, download, listed)


class PalomaTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)


class FetchTest(PalomaTest):
    def test_names_a_files_domain_after_the_split_or_by_its_source_when_the_source_is_one_domain(self):
        self.assertEqual(paloma.domain("dolma-v1_5", "dolma-v1_5/val/val_books.jsonl.gz"), "books")
        self.assertEqual(paloma.domain("c4_100_domains", "val_100_www.ign.com.jsonl.gz"), "100_www.ign.com")
        self.assertEqual(paloma.domain("ptb", "ptb/test/test.jsonl.gz"), "ptb")
        self.assertEqual(paloma.domain("c4_en", "c4_en/val/val-00000003.jsonl.gz"), "c4_en")

    def test_keeps_the_members_files_and_leaves_out_the_other_domains_of_a_source_that_mixes_kinds(self):
        texts = members_files()
        texts["dolma-v1_5/val/val_code.jsonl.gz"] = ["print(1)"]
        texts["c4_100_domains/val/val_1_www.example.com.jsonl.gz"] = ["a page", "another"]
        artifact = fetched(self.directory, texts)
        manifest = om.check(artifact)
        self.assertTrue(artifact.name.startswith("paloma-val-"))
        self.assertEqual((manifest["revision"], manifest["categories"]), (paloma.REVISION, CATEGORIES))
        held = paloma.files(artifact)
        self.assertIn(("dolma-v1_5", "books"), held)
        self.assertNotIn(("dolma-v1_5", "code"), held)
        self.assertEqual(list(paloma.documents(held["c4_100_domains", "1_www.example.com"])), ["a page", "another"])
        self.assertEqual(list(paloma.documents(held["ptb", "ptb"])), ["the text of ptb "])

    def test_refuses_a_source_that_gives_no_file_of_a_members_domain(self):
        texts = members_files()
        del texts["dolma-v1_5/val/val_books.jsonl.gz"]
        with self.assertRaisesRegex(ValueError, "books"):
            fetched(self.directory, texts)


class PartTest(PalomaTest):
    def test_gives_each_domain_the_beginnings_of_its_first_documents_within_its_bytes(self):
        texts = members_files()
        words = lambda count, word: " ".join(f"{word}{i:02d}" for i in range(count))  # four bytes a word, with its space
        # one domain, the whole member, in two files: 100 bytes over its three documents with text, so 33 each
        del texts["c4_en/val/val.jsonl.gz"], texts["c4_100_domains/val/val.jsonl.gz"]
        texts["c4_en/val/val-00000000.jsonl.gz"] = [words(10, "a"), "   ", words(2, "b")]
        texts["c4_en/val/val-00000001.jsonl.gz"] = [words(10, "c")]
        # a member of four domains: 100 bytes over them would be 25 each, and a domain gives no less than 30
        for name in "1234":
            texts[f"c4_100_domains/val/val_{name}.jsonl.gz"] = [f"{name}00"] * 20
        artifact = fetched(self.directory / "whole", texts)
        part = paloma.part(artifact, self.directory / "part", CATEGORIES, 100, 30, 10)
        manifest = om.check(part)
        self.assertTrue(part.name.startswith("paloma-val-sample-"))
        self.assertEqual((manifest["of"], manifest["member_bytes"]), (artifact.name, 100))
        held = paloma.files(part)
        self.assertEqual(set(held), set(paloma.files(artifact)))  # no domain is left out
        # each document from its beginning up to its share, cut after a space; the one without text is passed over
        self.assertEqual(list(paloma.documents(held["c4_en", "c4_en"])),
                         [words(8, "a") + " ", "b00 b01", words(8, "c") + " "])
        # documents shorter than the least of a document are whole, in their order, until the domain's bytes are full
        self.assertEqual(list(paloma.documents(held["c4_100_domains", "3"])), ["300"] * 10)
        self.assertEqual(paloma.part(artifact, self.directory / "again", CATEGORIES, 100, 30, 10).name, part.name)

    def test_passes_over_a_domain_without_text(self):
        texts = members_files()
        texts["dolma-v1_5/val/val_books.jsonl.gz"] = ["", "  "]  # as Paloma's test text has no books of Dolma's
        part = paloma.part(fetched(self.directory / "whole", texts), self.directory / "part", CATEGORIES, 100, 30, 10)
        self.assertNotIn(("dolma-v1_5", "books"), paloma.files(part))
        self.assertEqual(len(paloma.files(part)), len(paloma.members(CATEGORIES)) - 1)


class ReadTest(PalomaTest):
    def test_reads_a_document_whole_within_the_context_and_in_disjoint_pieces_beyond_it(self):
        self.assertEqual(paloma.pieces([0, 5, 6, 7], 8), [([0, 5, 6, 7], 3)])
        tokens = [0, *range(10, 20)]  # ten tokens after the BOS token
        rows = paloma.pieces(tokens, 4)
        # each piece begins with the last token of the one before, which it does not score again
        self.assertEqual(rows, [([0, 10, 11, 12, 13], 4), ([13, 14, 15, 16, 17], 4), ([17, 18, 19], 2)])
        self.assertEqual([token for row, scored in rows for token in row[-scored:]], tokens[1:])
        self.assertEqual(paloma.pieces([0], 4), [])  # a document without text has nothing to score

    def test_gives_each_domains_documents_rows_under_its_alias_whatever_files_hold_them(self):
        texts = members_files()
        del texts["c4_en/val/val.jsonl.gz"]
        texts["c4_en/val/val-00000000.jsonl.gz"] = ["ab", "c"]
        texts["c4_en/val/val-00000001.jsonl.gz"] = ["defgh"]
        artifact = fetched(self.directory, texts)
        encode = lambda batch: [list(text.encode("utf-8")) for text in batch]
        rows = dict(paloma.rows(artifact, encode, 0, 3))
        self.assertEqual(rows["paloma:c4_en/c4_en"],
                         [([0, 97, 98], 2), ([0, 99], 1), ([0, 100, 101, 102], 3), ([102, 103, 104], 2)])
        self.assertEqual(len(rows), len(paloma.members(CATEGORIES)))
        self.assertIn("paloma:dolma-v1_5/books", rows)

    def test_scores_bits_per_byte_from_the_negative_log_likelihood_over_the_texts_bytes(self):
        self.assertAlmostEqual(paloma.bits_per_byte(-8 * math.log(2), 4), 2.0)

    def test_counts_as_paragraphs_the_long_lines_without_the_space_around_them(self):
        long = "The quick brown fox jumps over the lazy dog and then it runs far away."
        text = f"A short line.\n   {long}  \n{'-- ' * 20}\n"
        self.assertEqual(list(paloma.paragraphs(text)), [long])  # not the short line, nor the line of marks
        self.assertEqual(len(paloma.digest(long)), 16)


if __name__ == "__main__":
    unittest.main()
