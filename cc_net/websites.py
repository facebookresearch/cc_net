import functools
import gzip
import itertools
import json
import logging
import multiprocessing
import requests
import subprocess
import urllib.parse
import zlib
from typing import Iterator, List, NamedTuple, BinaryIO
from pathlib import Path

import cc_net
import cc_net.process_wet_file
import submitit
from cc_net import jsonql

CC = "https://data.commoncrawl.org/"
INDEX_LIST_URL_PATTERN = CC + "crawl-data/CC-MAIN-{snapshot}/cc-index.paths.gz"

IDX = Path("/checkpoint/guw/hmine/indexes")
WARC = Path("/checkpoint/guw/hmine/warc")
FASTTEXT_MODEL = Path("")

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
    # TODO: download the indexes if needeed
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


SNAPSHOTS = "2022-05,2021-49,2021-43,2021-39,2021-31,2021-25,2021-21,2021-17,2021-10,2021-04,2020-50,2020-45,2020-40,2020-34,2020-29,2020-24,2020-16,2020-10,2020-05,2019-51,2019-47,2019-43,2019-39,2019-35,2019-30,2019-26,2019-22,2019-18,2019-13,2019-09,2019-04,2018-51,2018-47,2018-43,2018-39,2018-34,2018-30,2018-26,2018-22,2018-17,2018-13,2018-09,2018-05,2017-51,2017-47,2017-43,2017-39,2017-34,2017-30,2017-26,2017-22,2017-17,2017-13,2017-09,2017-04".split(
    ","
)


def find_in_cc(
    website: str, cache_file: Path = None, idx_dir: Path = IDX
) -> List[Document]:
    """Read all CC indexes to identify the documents from the given website.

    website: website to search for
    cache_file: a file that will be used to cache the results
    idx_dir: the directory to find the indexes
    """
    need_write = False

    if cache_file and cache_file.exists():
        docs = []
        for line in jsonql.open_read(cache_file):
            try:
                docs.append(Document.from_tsv(line))
            except AssertionError:
                # Fix error in some preexisting files
                docs.append(eval(line))
                need_write = True

        log.info(f"Loaded {len(docs)} docs from website {website}")
    elif cache_file and (WARC / "doc_lists" / f"{website}.docs.txt").exists():
        # reuse old naming schem
        cache_file.parent.mkdir(exist_ok=True)
        (WARC / "doc_lists" / f"{website}.docs.txt").rename(cache_file)
        return find_in_cc(website, cache_file, idx_dir)
    else:
        need_write = True
        docs = []
        for snapshot in SNAPSHOTS:
            docs.extend(find_in_snapshot(website, snapshot, idx_dir))

        log.info(f"Found {len(docs)} docs from website {website}")

    if cache_file and need_write:
        with jsonql.tmp_outfile(cache_file) as o:
            for doc in docs:
                print(*doc, sep="\t", file=o)
    return docs


def dl_from_cc(website: str, outfile: Path, idx_dir: Path = IDX) -> None:
    assert outfile.suffix == ".gz"

    text_outfile = outfile.parent / outfile.stem
    if text_outfile.exists():
        recompress_warc_file(text_outfile, outfile)

    doclist_file = outfile.parent / "doc_lists" / f"{website}.docs.txt"
    docs = find_in_cc(website, doclist_file, idx_dir)

    progress_file = outfile.with_suffix(".progress")
    if progress_file.exists():
        progress = progress_file.read_text().splitlines()
        log.info(f"Resuming from {len(progress)} downloaded pages in {outfile}")
    else:
        progress = []

    # Note: first I thought I could optimize the download by grouping downloads
    # from the same warc files. But unfortunately even for a given website,
    # it's rare to find two pages in the same warc file.
    n_docs = len(docs)
    optimization_potential = 0
    last_url = ""
    for doc in sorted(docs):
        if doc.url == last_url:
            optimization_potential += 1
        last_url = doc.url
    log.info(f"optimization_potential: {optimization_potential / n_docs:.2%}")

    with open(outfile, "ab") as o, open(progress_file, "a") as progress_o:
        for i, (url, doc) in enumerate(itertools.zip_longest(progress, docs)):
            if url is not None:
                assert url == doc.url
                continue
            dl_doc(doc, o)
            print(doc.url, file=progress_o, flush=True)
            log.info(f"Downloaded {i+1}/{n_docs} pages for website {website}")


def dl_batch(websites: List[str], outdir: Path, idx_dir: Path = IDX):
    n = len(websites)
    cached, downloaded, failed = 0, 0, 0
    for i, website in enumerate(websites):
        log.info(
            f"Downloaded {downloaded}/{n} websites. Failed {failed}, {cached} found on disks"
        )
        if (outdir / website).exists():
            cached += 1
            downloaded += 1
            continue
        try:
            dl_from_cc(website, outdir / (website + ".gz"), idx_dir)
            downloaded += 1
            log.info(f"Downloaded {website}")
        except Exception as e:
            log.error(f"Error while downloading {website}")
            log.exception(e)
            failed += 1


def dl_doc(doc: Document, outfile: BinaryIO):
    start, end = doc.offset, doc.offset + doc.length - 1
    segment_bytes = jsonql.request_get_content(
        CC + doc.segment, headers={"Range": f"bytes={start}-{end}"}
    )
    try:
        # Note: the input is gzip compressed, don't try to uncompress it,
        # just copy the bytes over to the file
        outfile.write(segment_bytes)
        outfile.flush()
    except Exception:
        log.error(f"Error while downloading {doc}")
        raise


CIRRUS = Path("/private/home/guw/github/cirrus-scripts")


def recompress_warc_file(warc_file: Path, warc_file_gz: Path = None) -> Path:
    warc_file_gz = warc_file_gz or Path(str(warc_file) + ".gz")
    if warc_file_gz.exists():
        return warc_file_gz

    n_docs = 0
    warc_file_gz_tmp = jsonql._tmp(warc_file_gz)
    o = gzip.open(warc_file_gz_tmp, "wt")
    follow_empty_line = False
    for line in jsonql.open_read(warc_file):
        if follow_empty_line and line == "WARC/1.0\n":
            o.close()
            o = gzip.open(warc_file_gz_tmp, "at")
            n_docs += 1
        follow_empty_line = len(line) == 1
        o.write(line.rstrip("\n"))
        o.write("\r\n")
    o.close()
    warc_file_gz_tmp.rename(warc_file_gz)
    log.info(f"Recompressed {warc_file_gz}, found {n_docs} documents")
    warc_file.unlink()
    return warc_file_gz


def extract_lett(warc_file: Path, fasttext_model: Path, outfile: Path) -> dict:
    """Lett files are tsv files with the following columns:

    language, metadata1, metadata2, url, base64 encoded text document
    """
    if not warc_file.suffix == ".gz":
        # Necessary for files that I manually compressed
        warc_file = recompress_warc_file(warc_file)

    outfile = outfile or warc_file.with_suffix(".lett.gz")

    tmp = jsonql._tmp(outfile)
    cmd = [
        # Note: this works because I'm using the gwenzek fork of warc2text that produces lett files
        CIRRUS / "bin" / "warc2text",
        "--tag-filters",
        CIRRUS / "mt-filter-list.annotated",
        "--url-filters",
        CIRRUS / "url-filter-list.optimised",
        "--langid-model",
        fasttext_model,
        "--output",
        tmp,
        warc_file,
    ]
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)
    tmp.rename(outfile)

    return lett_file_stats(outfile)


def lett_file_stats(lett_file: Path) -> dict:
    stats_txt = subprocess.check_output(f"zcat {lett_file} | cut -f1 | sort | uniq -c", shell=True, text=True)
    lines = [line.strip().split(" ", 1) for line in stats_txt.splitlines()]
    return {lang: int(n) for n, lang in lines}


if __name__ == "__main__":
    # docs = [
    #     Document(*x)
    #     for x in json.loads((WARC / "williamsfoodservice.co.uk.docs.json").read_text())
    # ]
    # website = "liriklagurinalpurba.blogspot.com"
    # dl_from_cc(website, WARC / "mri" / website)

    lang, website = "tir", "028baitong.com"
    # website = "bakanyu.com"
    dl_from_cc(website, WARC / lang / (website + ".gz"))
    FASTTEXT_MODEL = Path("/large_experiments/nllb/mmt/lidruns/lid_models/2022-02-18_ft_model.bin")
    lett = WARC.parent / "lett" / lang / (website + ".gz")
    # extract_lett(WARC / lang / (website + ".gz"), FASTTEXT_MODEL, lett)
    print(lett_file_stats(lett))
