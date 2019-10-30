# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json

from cc_net import dedup, jsonql
from cc_net.dedup import str_hash
from cc_net.flat_hash_set import FlatHashSet
from tests.test_jsonql import TestCaseWithTmpDir


def text(*args):
    return "\n".join(args)


def write_docs(filename, docs):
    with open(filename, "w") as f:
        for sentences in docs:
            doc = dict(text=text(*sentences))
            print(json.dumps(doc), file=f)


def dump_hashes(file, hashes):
    h = dedup.FlatHashSet()
    h.add(hashes)
    h.dump(file)


def as_dict(hash_set):
    if not isinstance(hash_set, dict):
        hash_set = {k: v for (k, v) in hash_set.items()}
    return hash_set


def load_hashes(file):
    results = dedup.FlatHashSet()
    results.load(file)
    return as_dict(results)


CUMBERSOME = ["original_length", "length"]


def assert_documents_equal(expected, actual, ignoring={}):
    expected = [{k: doc[k] for k in doc if k not in ignoring} for doc in expected]
    actual = [{k: doc[k] for k in doc if k not in ignoring} for doc in expected]
    assert expected == actual


def test_simple_dedup():
    documents = [
        dict(text=text("_Hello", "_World", "I'm so original")),
        dict(text=text("_world", "I'm originaler", "_Hello")),
    ]

    results = list(dedup.deduplicate(documents, field="text"))
    expected = [
        # First document is untouched
        dict(
            text=text("_Hello", "_World", "I'm so original"),
            original_nlines=3,
            nlines=3,
            text_hash=[str_hash(h) for h in ["_hello", "_world", "i'm so original"]],
        ),
        # Second documents loses several lines
        dict(
            text="I'm originaler",
            original_nlines=3,
            nlines=1,
            text_hash=[0, str_hash("i'm originaler"), 0],
        ),
    ]

    assert_documents_equal(expected, results, ignoring=CUMBERSOME)


class DedupTest(TestCaseWithTmpDir):
    def test_dedup_with_dump(self):
        tmp = self.get_tmpdir()

        documents = [
            dict(text=text("_Hello", "_World", "I'm so original")),
            dict(text=text("_world", "I'm originaler", "_Hello")),
        ]
        list(
            dedup.deduplicate(documents, field="text", output_hashes=tmp("hashes.bin"))
        )
        results = load_hashes(tmp("hashes.bin"))
        expected = {
            str_hash(l): l.startswith("_")
            for l in ["_hello", "_world", "i'm so original", "i'm originaler"]
        }
        self.assertEqual(expected, results)

    def test_dedup_with_np_dump(self):
        tmp = self.get_tmpdir()

        documents = [
            dict(text=text("_Hello", "_World", "I'm so original")),
            dict(text=text("_world", "I'm originaler", "_Hello")),
        ]
        with dedup.HashesCollector(field="text", output=tmp("hashes.bin")) as d:
            list(d.map(documents))

        results = FlatHashSet()
        results.load_np(tmp("hashes.bin"))
        expected = set(
            str_hash(l)
            for l in ["_hello", "_world", "i'm so original", "i'm originaler"]
        )
        self.assertEqual(expected, set(results.keys()))

    def test_dedup_with_hashes(self):
        tmp = self.get_tmpdir()

        documents = [
            dict(text=text("_Hello", "World", "I'm so original")),
            dict(text=text("Good morning", "World", "I'm originaler")),
        ]
        dump_hashes(
            tmp("hashes.bin"), [str_hash(h) for h in ["_hello", "i'm originaler"]]
        )
        results = list(
            dedup.deduplicate(
                documents, field="text", hashes=tmp("hashes.bin"), add_hashes=False
            )
        )
        expected = [
            dict(
                text=text("World", "I'm so original"),
                original_nlines=3,
                nlines=2,
                text_hash=[0, str_hash("world"), str_hash("i'm so original")],
            ),
            dict(
                text=text("Good morning", "World"),
                original_nlines=3,
                nlines=2,
                text_hash=[str_hash("good morning"), str_hash("world"), 0],
            ),
        ]

        assert_documents_equal(expected, results, ignoring=CUMBERSOME)

    def test_dedup_fast(self):
        data = self.get_tmpdir()
        part_0 = [["Hello", "_World", "I'm so original"]]
        write_docs(data("part_0.json"), part_0)
        part_1 = [["Good morning", "_World", "I'm originaler"]]
        write_docs(data("part_1.json"), part_1)

        res = self.get_tmpdir()
        h = self.get_tmpdir()
        dedup.deduplicate_concatenated(
            [data("part_0.json"), data("part_1.json")],
            [res("part_0.json"), res("part_1.json")],
            field="text",
            output_hashes=h("hashes.bin"),
        )

        with open(res("part_0.json")) as o:
            results_0 = [json.loads(l) for l in o.readlines()]
        expected_0 = [
            dict(
                text=text("Hello", "_World", "I'm so original"),
                original_nlines=3,
                nlines=3,
                text_hash=[str_hash(w) for w in ["hello", "_world", "i'm so original"]],
            )
        ]
        assert_documents_equal(expected_0, results_0, ignoring=CUMBERSOME)

        with open(res("part_1.json")) as o:
            results_1 = [json.loads(l) for l in o.readlines()]
        expected_1 = [
            dict(
                text=text("Good morning", "I'm originaler"),
                original_nlines=3,
                nlines=2,
                text_hash=[str_hash("good morning"), 0, str_hash("i'm originaler")],
            )
        ]

        assert_documents_equal(expected_1, results_1, ignoring=CUMBERSOME)

        words = [w for part in [part_0, part_1] for doc in part for w in doc]
        expected = {str_hash(s.lower()): s.startswith("_") for s in words}
        self.assertEqual(expected, load_hashes(h("hashes.bin")))

    def test_remove_duplicates_sharded(self):
        data = self.get_tmpdir()
        part_0 = [["Hello", "_World", "I'm so original"]]
        write_docs(data("part_0.json"), part_0)
        part_1 = [["_Good morning", "_World", "I'm originaler"]]
        write_docs(data("part_1.json"), part_1)

        h = self.get_tmpdir()
        h0 = FlatHashSet()
        h0.add([str_hash(s.lower()) for doc in part_0 for s in doc])
        h0.add([str_hash("_world")])
        h0.dump(h("part_0.bin"))
        self.assertEqual(
            {
                str_hash("hello"): False,
                str_hash("_world"): True,
                str_hash("i'm so original"): False,
            },
            as_dict(h0),
        )

        h1 = FlatHashSet()
        h1.add([str_hash(s.lower()) for doc in part_1 for s in doc])
        h1.add([str_hash("_good morning")])
        h1.dump(h("part_1.bin"))
        self.assertEqual(
            {
                str_hash("_good morning"): True,
                str_hash("_world"): False,
                str_hash("i'm originaler"): False,
            },
            as_dict(h1),
        )

        res = self.get_tmpdir()
        # dedup.DISABLE_MULTI_PROCESSING = True  # Simplifies debugging
        dedup.remove_duplicates_sharded(
            files=[data("part_0.json"), data("part_1.json")],
            outputs=[res("part_0.json"), res("part_1.json")],
            field="text",
            hashes_dir=h(),
        )

        with open(res("part_0.json")) as o:
            lines = o.readlines()
            print(lines)
            results_0 = list(jsonql.read_jsons(lines))
        expected_0 = [
            dict(text=text("Hello", "I'm so original"), original_nlines=3, nlines=2)
        ]
        assert_documents_equal(expected_0, results_0, ignoring=CUMBERSOME)

        with open(res("part_1.json")) as o:
            results_1 = [json.loads(l) for l in o.readlines()]
        # First pass removes "_world", second "_good morning".
        expected_1 = [dict(text=text("I'm originaler"), original_nlines=3, nlines=1)]

        assert_documents_equal(expected_1, results_1, ignoring=CUMBERSOME)
