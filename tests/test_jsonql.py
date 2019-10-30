# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import io
import os
import tempfile
import unittest
from itertools import zip_longest
from pathlib import Path
from typing import List

import numpy as np

from cc_net import jsonql


def bar(small_bar):
    return small_bar.replace(" ", " " * 10).replace("█", "█" * 10)


def get_output(transformer, data, **kwargs):
    with io.StringIO() as output:
        # Convert data to a generator so that it's not interpreted as a file list.
        jsonql.run_pipe(transformer, kwargs, file=(x for x in data), output=output)
        return output.getvalue()


class TestCaseWithTmpDir(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self._tmp_dirs: List[Path] = []

    def ignoring(self, dict_keys=None):
        dict_keys = dict_keys or []

        def dict_eq(expected, result, msg):
            r = result.copy()
            for f in dict_keys:
                r.pop(f, None)
            self.assertDictEqual(expected, r, msg)

        return self.with_dict_eq(dict_eq)

    def focusing_on(self, dict_keys):
        dict_keys = set(dict_keys)

        def dict_eq(expected, result, msg):
            self.assertLessEqual(expected.keys(), dict_keys)
            r = {k: result[k] for k in expected}
            self.assertDictEqual(expected, r, msg)

        return self.with_dict_eq(dict_eq)

    @contextlib.contextmanager
    def with_dict_eq(self, dict_eq):
        def list_eq(expected, results, msg):
            for e, r in zip_longest(expected, results):
                self.assertEqual(e, r, msg)

        # Default list equality uses "==" instead of "assertEqual", we need to
        # override it to be able to compare list of dicts with `dict_eq`.
        self.addTypeEqualityFunc(list, list_eq)
        self.addTypeEqualityFunc(dict, dict_eq)
        try:
            yield None
        finally:
            self.addTypeEqualityFunc(dict, None)

    def tearDown(self):
        for tmp_dir in self._tmp_dirs:
            tmp_dir.__exit__(None, None, None)
        self._tmp_dirs = []

    def get_tmpdir(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_dirs.append(tmp_dir)

        def tmp(*basename):
            return Path(os.path.join(tmp_dir.name, *basename))

        return tmp


class JsonqlTest(TestCaseWithTmpDir):
    def test_split(self):
        tmp = self.get_tmpdir()
        data = [
            dict(text="Hello world", lang="en"),
            dict(text="Boujour les amis", lang="fr"),
            dict(text="Rock your boat", lang="en"),
        ]
        with jsonql.split(tmp("{lang}.json")) as split:
            list(split.map(data))
            summary = split.summary()
        self.assertIn("Found 2 splits.", summary)
        with open(tmp("en.json")) as f_en:
            en_docs = list(jsonql.read_jsons(f_en))
            self.assertEqual([data[0], data[2]], en_docs)

        with open(tmp("fr.json")) as f_fr:
            fr_docs = list(jsonql.read_jsons(f_fr))
            self.assertEqual([data[1]], fr_docs)

    def test_split_bad_pattern(self):
        tmp = self.get_tmpdir()
        data = [dict(text="Hello world", lang="en")]
        with self.assertRaises(KeyError):
            with jsonql.split(tmp("{language}.json")) as split:
                list(split.map(data))

    def test_histogram(self):
        data = [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9]
        hist, bins = jsonql.histogram(data, bins=8, weights=None)
        np.testing.assert_almost_equal(bins, [0.1 * x for x in range(1, 10)])
        np.testing.assert_almost_equal(hist, [4, 0, 0, 2, 0, 0, 0, 2])

        data = [0, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.8, 0.8, 1]
        hist, bins = jsonql.histogram(data, bins=10, weights=None)
        np.testing.assert_almost_equal(bins, [0.1 * x for x in range(11)])
        np.testing.assert_almost_equal(hist, [1, 4, 0, 0, 2, 0, 0, 0, 2, 1])

    def test_display_stats(self):
        stats = {
            jsonql.ALL_DOCUMENTS: 100,
            "title": 80,
            "title.length": 80 * 50,
            "text": 100,
            "text.length": 100 * 1000,
            "popularity": 8,
            "popularity.val": [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9],
        }

        (title,) = jsonql.display_stats(stats, "title")
        self.assertIn("title", title)
        self.assertIn("saw 80 times", title)
        self.assertIn("average length is", title)
        self.assertNotIn("\n", title)

        (text,) = jsonql.display_stats(stats, "text")
        self.assertIn("text", text)
        self.assertIn("saw 100 times", text)
        self.assertIn("average length is", text)
        self.assertNotIn("\n", text)

        histogram = jsonql.display_stats(
            stats, "popularity", bins=[x / 10 for x in range(1, 10)]
        )
        self.assertIn("popularity", histogram[0])
        self.assertIn("saw 8 times", histogram[0])
        self.assertIn("histogram is", histogram[0])
        self.assertIn("0.100 " + bar("████████"), histogram[1])
        self.assertIn("0.400 " + bar("████    "), histogram[2])
        self.assertIn("0.800 " + bar("████    "), histogram[3])

        cum_histogram = jsonql.display_stats(
            stats, "popularity", bins=8, cumulative=True
        )
        self.assertIn("popularity", cum_histogram[0])
        self.assertIn("saw 8 times", cum_histogram[0])
        self.assertIn("histogram is", cum_histogram[0])
        self.assertIn("0.100 " + bar("████    "), cum_histogram[1])
        self.assertIn("0.400 " + bar("██████  "), cum_histogram[2])
        self.assertIn("0.800 " + bar("████████"), cum_histogram[3])

    def test_describe(self):
        def sample(pop):
            return dict(
                title="Lorem", text="Lorem ipsum dolor sit amet.", popularity=pop
            )

        data = [sample(pop) for pop in [0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.9]]
        desc = get_output(
            jsonql.describe, data, columns=None, bins=[x / 10 for x in range(1, 10)]
        )

        self.assertIn("Field title saw 8 times (100.0%), average length is 5", desc)
        self.assertIn("Field text saw 8 times (100.0%), average length is 27", desc)
        self.assertIn("Field popularity saw 8 times (100.0%), histogram is", desc)
        self.assertIn("0.100 " + bar("████████"), desc)
        self.assertIn("0.400 " + bar("████    "), desc)
        self.assertIn("0.800 " + bar("████    "), desc)

        desc = get_output(jsonql.describe, data, columns=["text"])
        self.assertNotIn("Field title saw 8 times (100.0%), average length is 5", desc)
        self.assertIn("Field text saw 8 times (100.0%), average length is 27", desc)
        self.assertNotIn("Field popularity, histogram is:", desc)


class JsonqlUtilsTest(TestCaseWithTmpDir):
    def test_custom_pipe(self):
        def transformer(source, sep=" "):
            for i, line in enumerate(source):
                res = f"{i}{sep}{line}"
                yield res

        data = ["hello", "world"]
        self.assertEqual(get_output(transformer, data), "0 hello\n1 world\n")
        self.assertEqual(get_output(transformer, data, sep="_"), "0_hello\n1_world\n")

    def test_smart_open(self):
        tmp = self.get_tmpdir()

        def readlines(filename):
            with jsonql.smart_open(filename) as f:
                return list(jsonql.lines(f))

        with jsonql.smart_open(tmp("a.txt"), "w") as o:
            print("a", file=o)
        self.assertEqual(readlines(tmp("a.txt")), ["a"])

        # with jsonql.smart_open(tmp("a.json.gz"), "w") as o:
        #     print("a", file=o)
        # self.assertEqual(readlines(tmp("a.json.gz")), ["a"])

        with jsonql.smart_open([tmp("a0.txt"), tmp("a1.txt")], "w") as o:
            print("a", file=o)
        self.assertEqual(readlines(tmp("a0.txt")), ["a"])
        self.assertFalse(os.path.isfile(tmp("a1.txt")))

        with jsonql.smart_open([tmp("b0.txt"), tmp("b1.txt")], "w", max_size="1k") as o:
            print("0" * 2000, file=o)
            print("1" * 2000, file=o)
        self.assertEqual(readlines(tmp("b0.txt")), ["0" * 2000])
        self.assertEqual(readlines(tmp("b1.txt")), ["1" * 2000])

        with jsonql.smart_open(tmp("a_????.json"), "w") as o:
            print("a", file=o)
        self.assertEqual(readlines(tmp("a_0000.json")), ["a"])
        self.assertFalse(os.path.isfile(tmp("a_0001.json")))
        self.assertEqual(readlines(tmp("a_*.json")), ["a"])

        with jsonql.smart_open(tmp("b_??.json"), "w", max_size="1k") as o:
            print("0" * 2000, file=o)
            print("1" * 2000, file=o)
        self.assertEqual(readlines(tmp("b_00.json")), ["0" * 2000])
        self.assertEqual(readlines(tmp("b_01.json")), ["1" * 2000])
        self.assertEqual(readlines(tmp("b_*.json")), ["0" * 2000, "1" * 2000])

    def test_split_file(self):
        tmp = self.get_tmpdir()
        file = tmp("test.txt")
        content = "Hello\nWorld\n"

        with open(file, "w") as o:
            o.write(content)

        with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["Hello\n"])

        with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["World\n"])

    def test_split_file_middle_of_line(self):
        tmp = self.get_tmpdir()
        file = tmp("test.txt")
        content = "Hello _|_\nWorld\n"
        # split is here   ^

        with open(file, "w") as o:
            o.write(content)

        with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["Hello _|_\n"])

        with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["World\n"])

    def test_split_file_middle_of_char(self):
        tmp = self.get_tmpdir()
        file = tmp("test.txt")
        content = "Hello\U0001F40D\nWorld\n"
        # split is here       ^^

        with open(file, "w") as o:
            o.write(content)

        with jsonql.SplitFile(file, chunk=0, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["Hello🐍\n"])

        with jsonql.SplitFile(file, chunk=1, n_chunks=2) as f:
            self.assertEqual(f.readlines(), ["World\n"])


def test_blocked_gzip(tmp_path):
    file = tmp_path / "test.gz"
    # Each object is 10/11 bytes long. We have 2 of them by block.
    content = [f'{{"xx": {i}}}' for i in range(80)]
    with jsonql.BlockedGzipWriter(file, "wt", block_size="20B") as o:
        for line in content:
            print(line, file=o)

    with jsonql.JsonReader(strict=True) as jr:
        with jsonql.smart_open(file) as f:
            read_as_one_file = list(jr.map(f))

        expected = list(jr.map(content))
        assert expected == read_as_one_file

        with jsonql.smart_open(str(file) + "[0/40]") as f:
            reader = list(f)
        assert expected[:2] == list(jr.map(l for l in reader))

        with jsonql.smart_open(str(file) + "[39/40]") as f:
            reader = list(f)
        assert expected[-2:] == list(jr.map(l for l in reader))

        readers = jsonql.get_block_readers(file, 9)
        read_as_several_files = [list(jr.map(r)) for r in readers]
        # 40 splits of 2 docs, 9 readers -> 5 splits, 10 docs per reader
        assert list(jsonql.grouper(expected, 10)) == read_as_several_files


def test_enter_exit(capsys):
    class MyTransformer(jsonql.Transformer):
        def __enter__(self):
            print("trans: started")
            self.ready = True
            return self

        def __exit__(self, *args):
            print("trans: done")

        def do(self, x):
            return (x, x)

    def acc(values):
        print("acc: started")
        res = 0
        for (x, _) in values:
            res += int(x)
        print("acc: done")
        yield f"acc: result={res}"

    t = MyTransformer()
    data = (str(x) for x in range(10))
    print("pipeline: started")
    # Print to stdout.
    jsonql.run_pipes(t, acc, file=data)
    print("pipeline: done")
    out = capsys.readouterr().out
    assert (
        "\n".join(
            [
                "pipeline: started",
                "trans: started",
                "acc: started",
                "acc: done",
                f"acc: result=45",
                # Transformers are closed at the very end.
                "trans: done",
                "pipeline: done\n",
            ]
        )
        == out
    )


def test_write_to_stdout(capsys):
    lines = [str(x) for x in range(10)]
    jsonql.run_pipes(file=iter(lines))
    out = capsys.readouterr().out
    assert out == "\n".join(lines) + "\n"


def test_write_to_stdout_handle_newlines(capsys):
    lines = [str(x) + "\n" for x in range(10)]
    jsonql.run_pipes(file=iter(lines))
    out = capsys.readouterr().out
    assert out == "".join(lines)


def test_multiprocess(capsys):
    mult = jsonql.Mapper(lambda x: f"2x = {2 * int(x)}")
    jsonql.run_pipes(mult, processes=2, file=(str(x) for x in range(10)))
    out = set(capsys.readouterr().out.strip("\n").split("\n"))
    assert set(f"2x = {2 * x}" for x in range(10)) == out
