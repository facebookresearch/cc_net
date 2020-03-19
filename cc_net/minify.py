# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import base64
import hashlib
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union
from urllib.parse import urlparse

import numpy as np

from cc_net import jsonql
from cc_net.execution import get_executor
from cc_net.jsonql import mem_footprint_gb
from cc_net.process_wet_file import CCSegmentsReader

HASH_SIZE = 4
HASH_TYPE = np.uint32

PUBLIC_FIELDS = ["url", "digest"]
COMPUTED_FIELDS = ["cc_segment", "language", "language_score", "bucket", "perplexity"]
CC_NET_ROOT_FOLDER = "https://dl.fbaipublicfiles.com/cc_net/"
DATA = Path(__file__).parent.parent / "data"

# This is similar to dedup methods but with use 32 bits hashes.


def _b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)


def _str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return _b2i(h.digest())


def get_hashes(lines: Iterable[str]) -> List[bytes]:
    h = HASH_SIZE
    return [hashlib.sha1(bytes(l, encoding="utf-8")).digest()[:h] for l in lines]


def encode_hashes(hashes: Iterable[bytes]) -> str:
    return base64.b64encode(b"".join(hashes)).decode("ascii")


def encode_as_hashes(lines: Iterable[str]) -> str:
    return encode_hashes(get_hashes(lines))


def decode_hashes(compact: str) -> List[bytes]:
    all_hashes = base64.b64decode(compact)
    res = []
    assert len(all_hashes) % HASH_SIZE == 0
    for i in range(len(all_hashes) // HASH_SIZE):
        chunk = all_hashes[i * HASH_SIZE : (i + 1) * HASH_SIZE]
        res.append(chunk)

    return res


def get_doc_key(digest: str) -> int:
    assert digest.startswith("sha1:")
    h = base64.b32decode(digest[5:])
    return _b2i(h[:HASH_SIZE])


class Minifier(jsonql.Transformer):
    ready = True

    def __init__(self):
        self.fields = frozenset(COMPUTED_FIELDS + PUBLIC_FIELDS)
        self.collisions = 0

    def do(self, doc: dict) -> Optional[dict]:
        content = doc.pop("raw_content", None)
        if not content:
            return None
        hashes = get_hashes(content.split("\n"))
        hashes_set = set(hashes)
        self.collisions += len(hashes) - len(hashes_set)

        fields = self.fields
        keys = list(doc.keys())
        for k in keys:
            if k not in fields:
                doc.pop(k, None)
        doc["hashes"] = encode_hashes(hashes)
        p = doc.get("perplexity", 0)
        if p:
            doc["perplexity"] = round(p, 1)
        s = doc.get("language_score", 0)
        if s:
            doc["language_score"] = round(s, 2)
        return doc

    def summary(self):
        return [f"Found {self.collisions} collisions"]


class Unminifier(jsonql.Transformer):
    """Read back the text from CC dump for minified documents.

    CC dumps are split in segments. Each segment is 64Mb long.
    Unminifier uses two level of caching:
        1. Segments are saved to the disk after being downloaded the first time.
        2. All "interesting" documents read in a segment are kept in memory until
        they are read.
    """

    def __init__(self):
        self.ready = True
        self.metadata: Dict[int, dict] = {}
        self.mem_cache: Dict[int, dict] = {}

        self._segments: Set[str] = set()
        self.read_doc = 0
        self.missed_doc = 0
        self.missed_par = 0
        self.processed_par = 0

    def look_for(self, minified: Iterable[dict]) -> None:
        """Mark the given minified documents as "interesting".
        The matching document will be kept in memory when they are found.
        """
        for d in minified:
            key = get_doc_key(d["digest"])
            self.metadata[key] = d
            self._segments.add(d["cc_segment"])

    def do(self, doc: dict) -> Optional[dict]:
        digest = doc["digest"]
        key = get_doc_key(digest)
        if key not in self.metadata:
            return None

        metadata = self.metadata.pop(key)
        return self.clean(metadata, doc)

    def clean(self, doc: dict, full_doc: dict) -> Optional[dict]:
        hashes = set(_b2i(h) for h in decode_hashes(doc.pop("hashes")))
        content = full_doc["raw_content"]
        cleaned = []
        processed_par = 0
        for line in content.split("\n"):
            h = _str_hash(line)
            processed_par += 1
            if h not in hashes:
                continue
            # In case of in-document duplicate we keep only the first occurence.
            # TODO: only keep two occurences if the hash appears twice
            hashes.remove(h)
            cleaned.append(line)

        self.missed_par += len(hashes)
        self.processed_par += processed_par
        if not cleaned:
            self.missed_doc += 1
            return None

        doc["raw_content"] = "\n".join(cleaned)
        doc["title"] = full_doc["title"]
        doc["date_download"] = full_doc["date_download"]
        doc["original_length"] = full_doc["length"]
        doc["original_nlines"] = full_doc["nlines"]
        doc["length"] = len(doc["raw_content"])
        doc["nlines"] = len(cleaned)
        doc["source_domain"] = urlparse(doc["url"]).netloc
        return doc

    def summary(self) -> List[str]:
        summ = super().summary()
        mem = mem_footprint_gb()
        len_cache = len(self.mem_cache)
        if len_cache > 2000:
            breakpoint()
        summ.append(
            f"Read {self.read_doc:_}, stocking {len_cache:_} doc in {mem:.1f}g."
        )
        if self.missed_doc:
            r = self.missed_doc / self.processed
            summ.append(f"! Missed {self.missed_doc} documents ({r:.1%}) !")
        if self.missed_par:
            r = self.missed_par / self.processed_par
            summ.append(f"! Missed {self.missed_par} paragraphs ({r:.1%}) !")
        return summ


def _expand_files(files: List[Path]) -> List[Path]:
    if len(files) == 1 and files[0].is_dir():
        folder = files[0]
        files = sorted(folder.glob("*.json.gz"))
        print(f"Found {len(files)} files under {folder}/*.json.gz")
    assert files, "No files found"
    return files


def minify_file(file: Path, output: Path) -> str:
    """Minify the given file."""
    jsonql.run_pipes(Minifier(), file=file, output=output)
    return f"Minified {output}"


def minify(
    files: List[Path], output_dir: Path, execution: str = "mp", parallelism: int = -1
):
    """Minify all the files in the given folder."""
    files = _expand_files(files)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "files.txt", "w") as o:
        for f in files:
            print(f.name, file=o)
    outputs = [output_dir / f.name for f in files]
    ex = get_executor(
        "minify",
        output_dir / "logs",
        execution,
        timeout_hour=2,
        cpus=1,
        task_parallelism=parallelism,
    )
    ex(minify_file, files, outputs)


def unminify_file(file: Union[Path, str], output: Path, cache_dir: Path = None):
    unminifier = Unminifier()
    with jsonql.smart_open(file) as f:
        unminifier.look_for(jsonql.read_jsons(f))

    tmp = output.with_name("tmp." + output.name)
    cc = CCSegmentsReader(list(unminifier._segments), min_len=300, cache_dir=cache_dir)
    jsonql.run_pipes(unminifier, file=cc, output=tmp)
    tmp.rename(output)
    f_size = Path(file).stat().st_size if Path(file).exists() else 0
    o_size = output.stat().st_size
    mb = 1024 ** 2
    return f"Unminified {output} ({f_size // mb:_}Mb -> {o_size // mb:_}Mb)"


def unminify(
    files: List[str],
    output_dir: Path,
    execution: str = "mp",
    parallelism: int = -1,
    cache_dir: Path = None,
):
    """Minify all the files in the given folder."""
    if len(files) == 1 and Path(files[0]).is_dir():
        folder = Path(files[0])
        files = [str(f) for f in sorted(folder.glob("*.json.gz"))]
        print(f"Found {len(files)} files under {folder}/*.json.gz")

    assert len(files) > 0, "No files given."
    output_dir.mkdir(exist_ok=True)

    outputs = [output_dir / str(f).split("/")[-1] for f in files]
    if cache_dir is None:
        cache_dir = output_dir / "wet_cache"
    if str(cache_dir) == "none":
        cache_dir = None
    files = [f for f, o in zip(files, outputs) if not o.exists()]
    outputs = [o for o in outputs if not o.exists()]
    if not files:
        return
    ex = get_executor(
        "unminify",
        output_dir / "logs",
        execution,
        timeout_hour=8,
        cpus=1,
        task_parallelism=parallelism,
        mem_gb=32,
    )
    ex(unminify_file, files, outputs, itertools.repeat(cache_dir))


def select_urls(
    dump: str, languages: List[str] = None, bucket: str = "head"
) -> List[str]:
    urls = []
    languages_set = set(languages) if languages else None
    with jsonql.open_remote_file(CC_NET_ROOT_FOLDER + dump + "/files.txt") as f:
        for file in f:
            file = file.strip()
            lang, buck, shard = file.split(".")[0].split("_")
            if bucket != "all" and buck != "all" and bucket != buck:
                # File named "all_xx" means that language "xx" didn't have a LM
                continue
            if languages_set and lang not in languages_set:
                continue
            urls.append(CC_NET_ROOT_FOLDER + dump + "/" + file)
    return urls


def reproduce(
    language: List[str] = None,
    dump: str = "2019-09",
    bucket: str = "head",
    shard: str = None,
    output_dir: Path = DATA / "reproduce",
    execution: str = "mp",
    parallelism: int = -1,
    cache_dir: Path = None,
):
    """Reproduce paper results from official CC snapshot and precomputed results.

    - dump: CC dump id
    - language: languages to keep (defaults to all)
    - bucket: quality bucket ("head", "middle", "tail", "all")
        - head: highest quality according to wikipedia-trained LM
        - tail: lowest quality
        - all: get all buckets
        See paper for more details: https://arxiv.org/abs/1911.00359
        Languages without an LM, will have only one bucket "all" which is always
        downloaded.
    - shard: select one specific shard (format: {lang}_{bucket}_{id})
        see https://dl.fbaipublicfiles.com/cc_net/2019-09/files.txt for available shards
    - ouput_dir: output directory
    - execution: how to parallelize ("mp", "debug", "slurm", ...)
    - cache_dir: where the CC .wet files will be downloaded.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    if shard is not None:
        if not shard.endswith(".json.gz"):
            shard += ".json.gz"
        urls = [CC_NET_ROOT_FOLDER + dump + "/" + shard]
    else:
        urls = select_urls(dump, language, bucket)
    unminify(urls, output_dir / dump, execution, parallelism, cache_dir)


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(reproduce, minify_file, minify, unminify, unminify_file)
