# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Tools to remove duplicate paragraphs across one or several shards.
"""

import argparse
import gc
import hashlib
import logging
import multiprocessing
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Set, Union

import numpy as np

from cc_net import jsonql
from cc_net.flat_hash_set import HASH_TYPE, AbstractDedupHashSet, FlatHashSet
from cc_net.text_normalizer import normalize_for_dedup

BYTE_ORDER = "little"
HASH_SIZE = HASH_TYPE(0).nbytes
RUSAGE_UNIT = 1000 if platform.system() == "Linux" else 1
DISABLE_MULTI_PROCESSING = False

FilesOrDir = Union[List[Path], Path]


def get_args():
    parser = argparse.ArgumentParser(
        description="Read a set of json files and allow to query them",
        parents=[jsonql.io_parser()],
    )

    parser.add_argument("--field", type=str, default="raw_content")
    parser.add_argument("--output_hashes", type=str)
    parser.add_argument("--no_finalize", action="store_false", dest="finalize")
    # parser.add_argument("--mem_gb", type=int)
    parser.add_argument("--hashes", type=str)

    return vars(parser.parse_args())


def _b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)


def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return _b2i(h.digest())


def mem_footprint_gb(pid=None):
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / 1_000_000_000 * RUSAGE_UNIT


log = logging.getLogger(__name__).info


def run_par(processes):
    # This is different from multiprocessing.map since it allows for kwargs.
    processes = list(processes)
    if len(processes) == 1 or DISABLE_MULTI_PROCESSING:
        for f, args, kwargs in processes:
            f(*args, **kwargs)
        return

    log(f"Starting {len(processes)} subprocess")
    processes = [
        multiprocessing.Process(target=f, args=a, kwargs=kw) for (f, a, kw) in processes
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    failed = 0
    for p in processes:
        if p.exitcode != 0:
            log(f"Process failed with code {p.exitcode}: {p}")
            failed += 1
    assert failed == 0, f"{failed} processes failed..."


def split_file(file, n_splits):
    for i in range(n_splits):
        yield jsonql.SplitFile(file, i, n_splits)


def merge(hashes_1, hashes_2, output):
    if isinstance(hashes_1, str):
        h1 = FlatHashSet()
        h1.load(hashes_1)
    else:
        h1 = hashes_1

    if isinstance(hashes_2, str):
        h2 = FlatHashSet()
        h2.load(hashes_2)
    else:
        h2 = hashes_2

    h2_np = np.fromiter(h2.keys(), dtype=FlatHashSet.dtype, count=len(h2))
    dup = h1.__contains__(h2_np)

    # Dups between h1 and h2 will be set to 1, keys unique to h2 are copied to
    # h1 with their value.
    h1[h2_np] = dup
    if output:
        h1.dump(output)
    return h1


def merge_shard(hash_files, output):
    h = FlatHashSet()
    h.load(hash_files[0])
    for hash_file in hash_files[1:]:
        h = merge(h, hash_file, output=None)
        print(f"Merged {hash_file}. We now have {len(h)} hashes.")

    h.dump(output)
    print(f"Saved {len(h)} hashes to {output}.")


def _dump_sentence_hashes(source: Path, output: Path, field: str):
    treated = 0
    started = time.time()
    with jsonql.smart_open(source, "r") as f, open(output, "wb") as o:
        for doc in jsonql.read_jsons(f):
            content = doc.get(field)
            if not content:
                continue
            h = compute_hashes(content)
            if h is None:
                continue
            h.tofile(o)
            treated += 1
            if treated % 100_000 == 0:
                delay = time.time() - started
                log(
                    f"Computed {treated} documents hashes in {delay / 3600:.2f}h ({treated / delay} doc / s)"
                )


def _remove_duplicate_hashes(duplicates, source, output):
    batch_size = 100_000
    n_lines, n_lines_kept = 0, 0
    with open(source, "rb") as f, open(output, "wb") as o:
        log(f"Opening {source} with mode rb")
        log(f"Opening {output} with mode wb")
        while True:
            hashes = np.fromfile(f, dtype=HASH_TYPE, count=batch_size)
            if hashes.size == 0:
                break

            keep = duplicates[hashes] < 1
            kept = keep.sum()
            hashes *= keep
            hashes.tofile(o)

            n_lines += hashes.size
            n_lines_kept += kept

    removed = n_lines - n_lines_kept
    selectivity = n_lines_kept / n_lines if n_lines else 0
    log(f"Removed {removed} duplicate hashes with selectivity: {selectivity:3.1%}")


def remove_duplicates_sharded(
    files: List[Path],
    outputs: List[Path],
    hashes_dir: FilesOrDir,
    field: str,
    group_hashes: int = 1,
    tmp_dir: Path = None,
    min_len: int = 0,
):
    """Remove duplicates in several passes, when all hashes don't fit in RAM.

    Note: The current implementation is not doing a 'perfect' deduplication.
    If a hash appear exactly once in each shard of hashes it won't be detected
    as a duplicate. This can be fixed if hashes are fully dedup beforehand.
    """
    assert len(files) == len(outputs)

    if isinstance(hashes_dir, list):
        hashes_files = hashes_dir
    else:
        hashes_files = sorted(
            h for h in Path(hashes_dir).iterdir() if h.suffix == ".bin"
        )

    assert len(hashes_files) > 0, f"no hashes files found in: {hashes_dir}"

    if len(hashes_files) <= group_hashes:
        log("All hashes can be done in one pass, using DuplicatesRemover on", files)
        rm_dups = DuplicatesRemover(field, hashes_files)
        rm_dups._prepare()
        run_par(
            (jsonql.run_pipes, (rm_dups,), dict(file=f, output=o))
            for f, o in zip(files, outputs)
        )
        return

    log("Starting deduplicate_sharded on", files)
    tmp_directory = tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None)

    def tmp_files(i):
        return [
            Path(tmp_directory.name) / (f.name.split(".")[0] + f".{i}.bin")
            for f in files
        ]

    last = tmp_files(0)
    run_par((_dump_sentence_hashes, (f, tmp, field), {}) for f, tmp in zip(files, last))

    if isinstance(hashes_dir, list):
        hashes_files = hashes_dir
    else:
        hashes_files = sorted(
            h for h in Path(hashes_dir).iterdir() if h.suffix == ".bin"
        )
    for i, group in enumerate(jsonql.grouper(hashes_files, group_hashes)):
        hashes = FlatHashSet()
        for h in group:
            hashes.load(h)
            log(f"Loaded {h}, up to {len(hashes)} hashes ({mem_footprint_gb()}GB)")

        intermediates = tmp_files(i + 1)
        # Remove hashes in parallel. Since modern OS have "copy-on-write" and
        # `hashes` is read-only, we will only have one version of it in RAM.
        run_par(
            (_remove_duplicate_hashes, (hashes, f, tmp), {})
            for f, tmp in zip(last, intermediates)
        )
        # Force hashes to be freed, before we start allocating a new one.
        del hashes
        gc.collect()

        for tmp in last:
            os.remove(tmp)
        last = intermediates

    def finalize(source, dedup_hashes, min_len):
        n_chars, n_chars_kept = 0, 0
        with open(dedup_hashes, "rb") as hashes:
            for doc in jsonql.read_jsons(source):
                content = doc.get(field)
                if not content or len(content) < min_len:
                    continue
                sentences = content.split("\n")
                doc_hashes = np.fromfile(hashes, dtype=HASH_TYPE, count=len(sentences))
                chars, kept_chars = finalize_doc(doc, field, doc_hashes)
                n_chars += chars
                n_chars_kept += kept_chars
                yield doc
        selectivity = n_chars_kept / n_chars if n_chars else 0
        log(f"Kept {n_chars_kept} chars out of {n_chars} ({selectivity:.1%}).")

    dedup_hashes = last
    run_par(
        [
            (
                jsonql.run_pipe,
                (finalize,),
                dict(kwargs=dict(dedup_hashes=h, min_len=min_len), file=f, output=o),
            )
            for h, f, o in zip(dedup_hashes, files, outputs)
        ]
    )

    tmp_directory.cleanup()


def deduplicate_concatenated(files, outputs, field, output_hashes, finalize=True):
    """Deduplicate several files at once, using the same set of hashes for all."""
    hashes = FlatHashSet()
    dedup_kwargs = dict(
        field=field,
        hashes=hashes,
        add_hashes=True,
        output_hashes=None,
        finalize=finalize,
    )

    assert len(files) == len(outputs)
    for f, o in zip(files, outputs):
        jsonql.run_pipe(deduplicate, dedup_kwargs, file=f, output=o)
        log(f"Saw {len(hashes)} hashes.")

        if output_hashes:
            log(f"Dumping {len(hashes)} hashes to {output_hashes}.")
            hashes.dump(output_hashes)


def compute_hashes(content) -> Optional[np.ndarray]:
    if not content:
        return None
    lines = content.split("\n")
    # save hashes as bytes but reinterpret them as uint64.
    hashes = np.fromiter(
        (
            hashlib.sha1(bytes(normalize_for_dedup(l), encoding="utf-8")).digest()[
                :HASH_SIZE
            ]
            for l in lines
        ),
        dtype=np.dtype((bytes, HASH_SIZE)),
        count=len(lines),
    )
    return np.ndarray(dtype=HASH_TYPE, buffer=hashes.data, shape=hashes.shape)


def finalize_doc(doc, field, hashes=None):
    content = doc.get(field)
    lines = content.split("\n")
    n_chars = len(content)
    if "original_nlines" not in doc:
        doc["original_nlines"] = len(lines)
    if "original_length" not in doc:
        doc["original_length"] = n_chars
    if hashes is None:
        hashes = doc.pop(field + "_hash")

    # Remove duplicates inside doc
    seen: Set[int] = set()
    for i in range(len(hashes)):
        if hashes[i] in seen:
            hashes[i] = 0
        seen.add(hashes[i])
    new_lines = list(l for (l, h) in zip(lines, hashes) if h != 0)

    doc[field] = "\n".join(new_lines)
    doc["nlines"] = len(new_lines)
    n_chars_kept = len(doc[field])
    doc["length"] = n_chars_kept
    return n_chars, n_chars_kept


class HashesCollector(jsonql.Transformer):
    """
    Collect all hashes found of lines found in the `field` of the source documents.
    """

    parallelisable = False

    def __init__(
        self, field: str, output: Path = None, hashes: AbstractDedupHashSet = None
    ):
        super().__init__()
        self.n_lines = 0
        self.field = field
        self.output = output
        self.hashes = FlatHashSet() if hashes is None else hashes
        self.num_hashes_start = len(self.hashes)

    def summary(self) -> List[str]:
        summ = super().summary()
        h = (len(self.hashes) - self.num_hashes_start) // 1000
        max_mem = mem_footprint_gb()
        n = self.n_lines // 1000
        summ.append(
            f"Found {h:_}k unique hashes over {n:_} lines. Using {max_mem:.1f}GB of RAM."
        )
        return summ

    def do(self, doc: dict) -> None:
        doc_hashes = compute_hashes(doc.get(self.field))
        if doc_hashes is None:
            return
        self.hashes.add(doc_hashes)
        self.n_lines += doc_hashes.size

    def close(self):
        if self.output and self.hashes:
            self.hashes.dump(self.output)
            # Free up mem even if the transformer is kept somewhere else.
            self.hashes = FlatHashSet()


class DuplicatesRemover(jsonql.Transformer):
    """DuplicatesRemover"""

    # The hashes can't be pickled so they will have to be read back from disk.
    warn_when_pickling = True

    def __init__(self, field: str, hashes_files: List[Path]):
        super().__init__()
        self.field = field
        self.hash_field = field + "_hash"

        self.hashes_files = hashes_files
        self.duplicates: Optional[AbstractDedupHashSet] = None

        self.n_lines, self.n_lines_kept = 0, 0
        self.n_chars, self.n_chars_kept = 0, 0

    def _prepare(self):
        if self.duplicates is not None:
            return
        self.duplicates = FlatHashSet()

        start = time.time()
        for h in self.hashes_files:
            shard_start = time.time()
            self.duplicates.load(str(h))
            delay = time.time() - shard_start
            self.log(
                f"Loaded hashes from {h} ({mem_footprint_gb()}GB total, took {delay / 60:.1}m)"
            )

        delay = time.time() - start
        self.log(
            f"Loaded {len(self.duplicates)} hashes from {len(self.hashes_files)} files. ({mem_footprint_gb()}GB total, took {delay / 60:.1}m)"
        )

    def do(self, doc: dict) -> Optional[dict]:
        doc_hashes = doc.get(self.hash_field)
        if doc_hashes:
            doc_hashes = np.array(doc_hashes, dtype=HASH_TYPE)
        else:
            content = doc.get(self.field)
            if not content:
                return None
            doc_hashes = compute_hashes(content)

        assert self.duplicates is not None
        keep = self.duplicates[doc_hashes] < 1
        kept = keep.sum()
        if kept == 0:
            return None

        doc_hashes = doc_hashes * keep
        doc[self.hash_field] = [int(x) for x in doc_hashes]
        self.n_lines += keep.size
        self.n_lines_kept += kept

        chars, kept_chars = finalize_doc(doc, self.field)
        self.n_chars += chars
        self.n_chars_kept += kept_chars
        return doc

    def summary(self) -> List[str]:
        summ = super().summary()
        end_time = time.time()
        n_lines_kept, n_lines, n_docs = self.n_lines_kept, self.n_lines, self.processed
        speed = n_docs / (end_time - self.start_time)
        summ.append(
            f"Processed {self.n_lines} lines in {n_docs} docs. [{speed:.1f} doc/s]"
        )
        selectivity = self.n_lines_kept / self.n_lines if n_lines else 0
        summ.append(f"Kept {n_lines_kept} lines out of {n_lines} ({selectivity:.1%}).")

        n_chars_kept, n_chars = self.n_chars_kept, self.n_chars
        selectivity = n_chars_kept / n_chars if n_chars else 0
        summ.append(f"Kept {n_chars_kept} chars out of {n_chars} ({selectivity:.1%}).")
        return summ


def deduplicate(
    source, field, hashes=None, output_hashes=None, add_hashes=True, finalize=True
):
    """
    DOES TOO MANY THINGS
    Removes duplicate lines found in the field `field` of the source documents.

    Finds duplicate lines based on the hashes. Either hashes can be computed when
    reading the documents or they can be loaded from a binary file.

    If `add_hashes` is set to False only the given hashes will be considered.
    This grants a better control on memory footprint.
    """
    hash_field = field + "_hash"
    if isinstance(hashes, str) or isinstance(hashes, Path):
        seen = FlatHashSet()
        seen.load(hashes)
    elif hashes is not None:
        seen = hashes
    else:
        seen = FlatHashSet()
    log(f"Loaded {len(seen)} unique hashes.")
    n_doc = 0
    batch_size = 100_000
    n_lines, n_lines_kept = 0, 0
    n_chars, n_chars_kept = 0, 0
    t = time.time()

    def log_stats(start_time):
        end_time = time.time()
        speed = batch_size / (end_time - start_time)

        if add_hashes:
            log(
                f"Saw {len(seen)} unique hashes over {n_lines} lines in {n_doc} docs. [{speed:.1f} doc/s]"
            )
        else:
            log(f"Processed {n_lines} lines in {n_doc} docs. [{speed:.1f} doc/s]")
        max_mem = mem_footprint_gb()
        log(f"Used up to {max_mem:.1f}GB of RAM.")
        selectivity = n_lines_kept / n_lines if n_lines else 0
        log(f"Kept {n_lines_kept} lines out of {n_lines} ({selectivity:.1%}).")
        if finalize:
            selectivity = n_chars_kept / n_chars if n_chars else 0
            log(f"Kept {n_chars_kept} chars out of {n_chars} ({selectivity:.1%}).")

    for doc in jsonql.read_jsons(source):
        n_doc += 1
        if n_doc % batch_size == 0:
            log_stats(t)
            t = time.time()

        hashes = doc.get(hash_field) or compute_hashes(doc.get(field))
        if hashes is None:
            continue
        if isinstance(hashes, list):
            hashes = np.array(hashes, dtype=HASH_TYPE)

        duplicate = seen.__contains__(hashes)
        if add_hashes:
            seen.add(hashes, duplicate)

        keep = duplicate < 1
        kept = keep.sum()
        hashes = hashes * keep
        doc[hash_field] = list(int(x) for x in hashes)
        n_lines += keep.size
        n_lines_kept += kept
        if finalize:
            chars, kept_chars = finalize_doc(doc, field)
            n_chars += chars
            n_chars_kept += kept_chars
        if kept > 0:
            yield doc

    log_stats(t)

    if output_hashes:
        log(f"Dumping {len(seen)} hashes to {output_hashes}.")
        seen.dump(output_hashes)


def main():
    args = get_args()

    return jsonql.run_pipe(deduplicate, args)


if __name__ == "__main__":
    main()
