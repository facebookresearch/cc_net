import cc_net
import cc_net.process_wet_file

import requests
import multiprocessing
import functools
import submitit
import subprocess
import json
import urllib.parse
import logging
from typing import Iterator, List, NamedTuple, TextIO
from pathlib import Path
from cc_net import jsonql

CC = "https://data.commoncrawl.org/"
INDEX_LIST_URL_PATTERN = CC + "crawl-data/CC-MAIN-{snapshot}/cc-index.paths.gz"

IDX = Path("/checkpoint/guw/hmine/indexes")
WARC = Path("/checkpoint/guw/hmine/warc")

log = logging.getLogger("cc_net.websites")


def list_tasks(cache: Path):
    snapshots = cc_net.process_wet_file.list_dumps()[::-1]

    for snapshot in snapshots:
        year = int(snapshot.split("-")[0])
        if year < 2017:
            continue

        index_list = INDEX_LIST_URL_PATTERN.format(snapshot=snapshot)
        index_list_cache = cache / f"{snapshot}.list.gz"
        for line in jsonql.open_remote_file(index_list, cache=index_list_cache):
            line = line.rstrip()
            if not line.endswith(".gz"):
                continue
            yield (snapshot, line)


def dl_indexes(snapshot: str, outdir: Path) -> Path:
    outdir = outdir / snapshot
    outdir.mkdir(exist_ok=True)

    index_list = INDEX_LIST_URL_PATTERN.format(snapshot=snapshot)
    index_list_cache = outdir / f"{snapshot}.list.gz"
    for line in jsonql.open_remote_file(index_list, cache=index_list_cache):
        line = line.rstrip()
        if not line:
            continue

    return outdir


def dl_index(outdir: Path, snapshot: str, index_full_name: str) -> Path:
    url = CC + index_full_name
    name = index_full_name.split("/")[-1]
    output = (outdir / f"{snapshot}.{name}").with_suffix(".gz")
    if output.exists():
        return output
    tmp_output = output.with_suffix(".gztmp")

    print(f"downloading {url}")
    subprocess.check_call(["wget", "-q", "-O", str(tmp_output), url])
    tmp_output.rename(output)
    return output


class BtreeEntry(NamedTuple):
    filename: str
    rurl: str
    url: str


class IndexEntry(NamedTuple):
    rurl: str
    hash: int
    data: str


def read_idx(idx_file: Path) -> Iterator[IndexEntry]:
    for first_line in jsonql.open_read(idx_file):
        reversed_url, hash, data = first_line.strip().split(" ", 2)
        yield IndexEntry(reversed_url, int(hash), data)


def idx_btree(snapshot: str, idx_dir: Path = IDX) -> List[BtreeEntry]:
    idx_files = list(idx_dir.glob(f"{snapshot}.cdx-*.gz"))
    assert idx_files, f"Invalid snashot name {snapshot} for index dir: {idx_dir}"

    btree_dir = idx_dir / "idx_btree"
    btree_dir.mkdir(exist_ok=True)
    btree_file = btree_dir / f"{snapshot}.btree.json"

    if btree_file.exists():
        splits = [BtreeEntry(*e) for e in json.loads(btree_file.read_text())]
        if len(splits) == len(idx_files):
            return splits

    splits = []
    for idx in idx_files:
        for first_entry in read_idx(idx):
            data = json.loads(first_entry.data)
            splits.append(BtreeEntry(idx.name, first_entry.rurl, data["url"]))
            break

    splits.sort()
    print(splits)

    btree_file.write_text(json.dumps(splits, indent=2))
    return splits


def reverse_url(url_str: str):
    url = urllib.parse.urlparse(url_str)
    if ":" in url.netloc:
        host, port = url.netloc.split(":")
    else:
        host, port = url.netloc, ""
    rev_url = ",".join(host.split(".")[::-1])
    if port:
        rev_url += f":{port}"
    rev_url += f")/{url.path.strip('/')}"

    # if url.query:
    #     sep = ";" if ";" in url.query else "&"
    #     query = url.query.split(sep)
    #     query.sort()
    #     rev_url += "?" + sep.join(query)

    return rev_url.lower()


def test_reverse_url():
    assert (
        reverse_url(
            "http://orders.williamsfoodservice.co.uk:8080/apex/f?p=114:LOGIN_DESKTOP:24990902292733:::::"
        )
        == "uk,co,williamsfoodservice,orders:8080)/apex/f"
    )

    assert (
        reverse_url("http://5.135.0.0/2018/11/16/hej-varlden/")
        == "0,0,135,5)/2018/11/16/hej-varlden"
    )

    assert reverse_url("http://1.0/") == "0,1)/"

    # I gave up on url encoding
    # '0,4,163,85:9999)/faces/login.jsp;jsessionid=3ed4b846253a1bf6026865108035541a'
    assert (
        reverse_url(
            "http://85.163.4.0:9999/faces/login.jsp;jsessionid=3ED4B846253A1BF6026865108035541A"
        )
        == "0,4,163,85:9999)/faces/login.jsp"
    )
    # 0,203,62,116)/index.php?c=forumlist&m=bbs
    assert (
        reverse_url("http://116.62.203.0/index.php?m=bbs&c=forumlist")
        == "0,203,62,116)/index.php"
    )


class Document(NamedTuple):
    segment: str
    offset: int
    length: int
    url: str
    digest: str

    @staticmethod
    def from_json(json_dict: dict) -> "Document":
        return Document(
            segment=json_dict["filename"],
            offset=int(json_dict["offset"]),
            length=int(json_dict["length"]),
            url=json_dict["url"],
            digest=json_dict["digest"],
        )

    @staticmethod
    def from_tsv(line: str) -> "Document":
        parts = line.rstrip("\n").split("\t")
        assert len(parts) == 5
        return Document(
            segment=parts[0],
            offset=int(parts[1]),
            length=int(parts[2]),
            url=parts[3],
            digest=parts[4],
        )


def find_in_snapshot(
    website: str, snapshot: str, idx_dir: Path = IDX
) -> List[Document]:
    btree = idx_btree(snapshot)
    assert btree
    rurl = reverse_url("http://" + website).rstrip(")/")
    idx_file = idx_dir / btree[-1].filename
    for i, entry in enumerate(btree[1:]):
        if rurl < entry.rurl:
            idx_file = idx_dir / btree[i].filename
            break

    docs = []
    for idx_entry in read_idx(idx_file):
        if idx_entry.rurl < rurl:
            continue
        # Either idx_entry.rurl is a webpage from rurl, or it's another website
        if not idx_entry.rurl.startswith(rurl):
            # Since websites are sorted we can exit
            break

        doc = Document.from_json(json.loads(idx_entry.data))
        if "/warc/CC-MAIN-" not in doc.segment:
            # We don't want to read the crawldiagnostics and robottxt files
            continue
        docs.append(doc)

    return docs


# def dl_from_snapshot(
#     website: str, snapshot: str, outfile: TextIO, idx_dir: Path = IDX
# ) -> List[Document]:
#     docs = find_in_snapshot(website)
#     # breakpoint()
#     with open(doclist_file, "a") as doc_o:
#         for doc in docs:
#             print(doc, sep="\t", file=doc_o)
#     docs = merge_doc_ranges(docs)
#     # with jsonql.open_write(outdir / website) as o:
#     for doc in docs:
#         dl_doc(doc, outfile)

#     return docs


SNAPSHOTS = "2022-05,2021-49,2021-43,2021-39,2021-31,2021-25,2021-21,2021-17,2021-10,2021-04,2020-50,2020-45,2020-40,2020-34,2020-29,2020-24,2020-16,2020-10,2020-05,2019-51,2019-47,2019-43,2019-39,2019-35,2019-30,2019-26,2019-22,2019-18,2019-13,2019-09,2019-04,2018-51,2018-47,2018-43,2018-39,2018-34,2018-30,2018-26,2018-22,2018-17,2018-13,2018-09,2018-05,2017-51,2017-47,2017-43,2017-39,2017-34,2017-30,2017-26,2017-22,2017-17,2017-13,2017-09,2017-04".split(
    ","
)


def dl_from_cc(website: str, outfile: Path, idx_dir: Path = IDX) -> None:
    if outfile.exists():
        return

    doclist_file = outfile.parent / "doc_lists" / f"{website}.docs.txt"
    if doclist_file.exists():
        docs = []
        for line in jsonql.open_read(doclist_file):
            if len(line) <= 1:
                continue
            docs.append(Document.from_tsv(line))
        log.info(f"Loaded {len(docs)} docs from website {website}")
    elif (WARC / "doc_lists" / f"{website}.docs.txt").exists():
        doclist_file.parent.mkdir(exist_ok=True)
        breakpoint()
        (WARC / "doc_lists" / f"{website}.docs.txt").rename(doclist_file)
        return dl_from_cc(website, outfile, idx_dir)
    else:
        docs = []
        for snapshot in SNAPSHOTS:
            docs.extend(find_in_snapshot(website, snapshot, idx_dir))

        log.info(f"Found {len(docs)} docs from website {website}")

        with jsonql.tmp_outfile(doclist_file) as o:
            for doc in docs:
                print(doc, sep="\t", file=o)

    progress_file = Path(str(outfile) + ".progress")
    if progress_file.exists():
        progress = len(progress_file.read_text().splitlines())
        log.info(f"Resuming from {progress} downloaded pages in {outfile}")
    else:
        progress = 0

    with open(progress_file, "a") as progress_o:
        with jsonql.open_write(outfile) as o:
            for i, doc in enumerate(docs):
                if i < progress:
                    continue
                dl_doc(doc, o)
                print(doc.url, file=progress_o)


def dl_batch(websites: List[str], outdir: Path, idx_dir: Path = IDX):
    for website in websites:
        if (outdir / website).exists():
            continue
        try:
            dl_from_cc(website, outdir / website, idx_dir)
        except Exception as e:
            log.error(f"Error while downloading {website}")
            log.exception(e)


def merge_doc_ranges(docs: List[Document], ratio: float = 2.0) -> List[Document]:
    # TODO: it doesn't seems to help most of the time
    docs.sort()

    merged = []
    current_doc = docs[0]
    current_end = current_doc.offset + current_doc.length
    for i, doc in enumerate(docs[1:]):
        if current_doc.segment == doc.segment:
            if doc.offset - current_end < ratio * current_doc.length:
                current_doc = current_doc._replace(
                    length=doc.offset - current_doc.offset + doc.length
                )
                continue
        merged.append(current_doc)
        current_doc = doc
    # TODO: we should ask for list of range instead
    merged.append(current_doc)

    print(f"Merged {len(docs)} docs into {len(merged)} ranges")
    original_bytes = sum(d.length for d in docs)
    merged_bytes = sum(d.length for d in merged)
    print(f"Going from {original_bytes:_d} bytes to {merged_bytes:_d}")
    return merged


def dl_doc(doc: Document, outfile: TextIO):
    start, end = doc.offset, doc.offset + doc.length - 1
    # TODO: do we need to unzip the file right now ?
    # TODO: DON'T UNZIP AND KEEP DOCUMENT BOUNDARIES
    segment = jsonql.open_remote_file(
        CC + doc.segment, headers={"Range": f"bytes={start}-{end}"}
    )
    try:
        for line in segment:
            outfile.write(line)
    except Exception:
        log.error(f"Error while downloading {doc}")
        raise


CIRRUS = Path("/private/home/guw/github/cirrus-scripts")


def extract_text(warc_file: Path, outfile: Path = None):
    if not warc_file.suffix == ".gz":
        warc_file_gz = Path(str(warc_file) + ".gz")
        # TODO: recompress the files
        if not warc_file_gz.exists():
            subprocess.check_call(["gzip", warc_file])
        warc_file = warc_file_gz

    outfile = outfile or warc_file.with_suffix(".wet")

    tmp = jsonql._tmp(outfile)
    cmd = [
        CIRRUS / "bin" / "warc2text",
        "--tag-filters",
        CIRRUS / "mt-filter-list.annotated",
        "--url-filters",
        CIRRUS / "url-filter-list.optimised",
        "--output",
        tmp,
        "--multilang",
        warc_file,
    ]
    print(" ".join(str(x) for x in cmd))
    subprocess.run(
        cmd,
        check=True,
    )
    tmp.rename(outfile)


if __name__ == "__main__":
    # docs = [
    #     Document(*x)
    #     for x in json.loads((WARC / "williamsfoodservice.co.uk.docs.json").read_text())
    # ]
    # merge_doc_ranges(docs, ratio=10)
    # website = "liriklagurinalpurba.blogspot.com"
    # dl_from_cc(website, WARC / "mri" / website)

    # website = "bakanyu.com"
    website = "028baitong.com"
    extract_text(WARC / "tir" / website)
