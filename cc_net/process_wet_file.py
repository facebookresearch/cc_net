# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from urllib.parse import urlparse

import func_argparse
from bs4 import BeautifulSoup

from cc_net import jsonql

WET_URL_ROOT = "https://commoncrawl.s3.amazonaws.com"


logger = logging.getLogger(__name__)


def cc_segments_url(dump_id: str) -> str:
    return "/".join([WET_URL_ROOT, "crawl-data", "CC-MAIN-" + dump_id, "wet.paths.gz"])


def list_dumps() -> List[str]:
    home_page = BeautifulSoup(
        urllib.request.urlopen("http://index.commoncrawl.org/"), features="html.parser"
    )
    dumps = [a.get("href").strip("/") for a in home_page.findAll("a")]
    dumps = [a[8:] for a in dumps if re.match(r"^CC-MAIN-\d\d\d\d-\d\d$", a)]

    return sorted(dumps)


def ls():
    for dump in list_dumps():
        print(dump, "->", cc_segments_url(dump))


def parse_doc(headers: List[str], doc: List[str]) -> Optional[dict]:
    """Headers format is:
WARC/1.0
WARC-Type: conversion
WARC-Target-URI: [url]
WARC-Date: [crawldate: 2019-02-15T19:15:59Z]
WARC-Record-ID: <urn:uuid:8865156e-d5f1-4734-9c68-4b46eaf2bb7e>
WARC-Refers-To: <urn:uuid:340152e2-65cf-4143-b522-8ce4e2d069d7>
WARC-Block-Digest: sha1:S3DTWCONT2L6ORTGCY2KXEZ37LNBB7V2
Content-Type: text/plain
Content-Length: 7743
    """
    if not headers or not doc:
        return None

    try:
        warc_type = headers[1].split()[1]
        if warc_type != "conversion":
            return None
        url = headers[2].split()[1]
        date = headers[3].split()[1]
        digest = headers[6].split()[1]
        length = int(headers[8].split()[1])
    except Exception as e:
        logger.warning("Can't parse header:", e, headers, doc)
        return None

    title, doc = doc[0], doc[1:]
    return {
        "url": url,
        "date_download": date,
        "digest": digest,
        "length": length,
        "nlines": len(doc),
        "source_domain": urlparse(url).netloc,
        "title": title,
        "raw_content": "\n".join(doc),
    }


def group_by_docs(warc_lines: Iterable[str]) -> Iterable[dict]:
    doc: List[str] = []
    headers, read_headers = [], True
    for warc in warc_lines:
        warc = warc.strip()
        if read_headers:
            headers.append(warc)
            read_headers = warc != ""
            continue

        if warc == "WARC/1.0":
            # We reached the beginning of the new doc.
            parsed = parse_doc(headers, doc)
            if parsed is not None:
                yield parsed
            headers, doc, read_headers = [warc], [], True
            continue

        if warc:
            doc.append(warc)

    # Return the last document
    if doc:
        parsed = parse_doc(headers, doc)
        if parsed is not None:
            yield parsed


def parse_warc_file(lines: Iterable[str], min_len: int = 1) -> Iterator[dict]:
    n_doc = 0
    n_ok = 0
    for doc in group_by_docs(lines):
        n_doc += 1
        if not doc or len(doc["raw_content"]) < min_len:
            continue
        n_ok += 1
        yield doc
    if n_doc > 0:
        logger.info(f"Kept {n_ok:_d} documents over {n_doc:_d} ({n_ok / n_doc:.1%}).")
    else:
        logger.info(f"Found no documents")


def dl(
    dump: str,
    shard: int,
    num_shards: int,
    output: Path = None,
    num_segments_per_shard: int = 0,
):
    """Download a shard of the common crawl, and export it to json.

    Arguments:
        output: filename of the output file
        dump: CC dump id
        shard: id of the shard
        num_shards: total number of shards
        num_segments_per_shard: manual control of the number of segment per shard.
    """
    reader = CCShardReader(dump, shard, num_shards, num_segments_per_shard)
    jsonql.run_pipes(file=reader, output=output)
    logger.info(f"Done. {output} is ready.")


class CCShardReader(Iterable[dict]):
    def __init__(
        self,
        dump: str,
        shard: int,
        num_shards: int,
        num_segments_per_shard: int = -1,
        min_len: int = 0,
    ):
        """Downloads a shard of Common Crawl, and yields dict.

        Arguments:
            dump: CC dump id
            shard: id of the shard
            num_shards: total number of shards
            num_segments_per_shard: if set will limit the number of files by shard.
                Useful for testing.
        """
        self.dump = dump
        self.shard = shard
        self.num_shards = num_shards
        self.num_segments_per_shard = num_segments_per_shard
        self.min_len = min_len
        self._segments: List[str] = []

    @property
    def segments(self) -> List[str]:
        if self._segments:
            return self._segments
        segments_file = cc_segments_url(self.dump)
        with jsonql.smart_open(segments_file) as f:
            segments = [segment.strip() for segment in f]
        n = len(segments)
        i_min = (self.shard * n) // self.num_shards
        i_max = ((self.shard + 1) * n) // self.num_shards
        if self.num_segments_per_shard > 0:
            i_max = min(i_max, i_min + self.num_segments_per_shard)
        self._segments = segments[i_min:i_max]
        return self._segments

    def segment_url(self, segment: str):
        return "/".join((WET_URL_ROOT, segment))

    def __iter__(self) -> Iterator[dict]:
        n = len(self.segments)
        for i, segment in enumerate(self.segments):
            start = time.time()
            # TODO: start downloading the next segment in the background
            with jsonql.open_remote_file(self.segment_url(segment)) as f:
                for doc in parse_warc_file(iter(f), self.min_len):
                    doc["cc_segment"] = segment
                    yield doc

            if i + 1 >= n:
                continue
            end = time.time()
            delay = (end - start) / 3600 * (n - 1 - i)
            logger.info(
                f"Parsed {i + 1} / {n} files. Estimated remaining time: {delay:.1f}h"
            )


if __name__ == "__main__":
    func_argparse.main(ls, dl)
