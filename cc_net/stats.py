import base64
import collections
import functools
import glob
import itertools
import logging
import lzma
import multiprocessing
import os
import pickle
import subprocess
import sys
import urllib.parse
import zlib
from typing import Counter, Dict, List, Iterable, Optional, Set, Tuple
from pathlib import Path

import func_argparse

import cc_net.dedup
import cc_net.regroup
import cc_net.execution
import lzma
import submitit
from collections import defaultdict
from cc_net import jsonql

log = logging.getLogger("stats")

PUBLIC_SUFFIX_LIST = Path(__file__).parent / "data" / "public_suffix_list.dat"


def long_running(fn):
    if "screen" in os.environ["TERM"]:
        return fn

    @functools.wraps(fn)
    def restart_from_screen(*args, **kwargs):
        breakpoint()
        subprocess.check_call(["screen", "-R", fn.__name__, sys.executable] + sys.argv)

    return restart_from_screen


class DomainShortener:
    """
    Extract good domain names from URLs.

    Uses a list of public suffixes that don't indicate a real domain name
    eg: "blogpost.com", ".gov.uk", ".co.in", ...
    """

    def __init__(self):
        self.suffixes: Set[str] = set()
        self.meta_suffixes: Set[str] = set()
        self._prepare()

    def _prepare(self):
        lines = PUBLIC_SUFFIX_LIST.read_text().splitlines()
        for line in lines:
            domain = line.strip()
            if not domain or domain.startswith("//"):
                # Skip comments and blank lines
                continue
            if domain.startswith("!"):
                # Those are exceptions to the rules, so not respecting them
                # means we are over-splitting. Given the few exceptions
                # I don't think it's a huge problem.
                continue

            if domain.startswith("*."):
                self.meta_suffixes.add(domain[2:])
                continue

            i = len(domain)
            while i >= 0:
                i = domain.rfind(".", 0, i)
                suffix = domain[i + 1 :]
                self.suffixes.add(suffix)

    def __call__(self, domain):
        i = len(domain)
        while i >= 0:
            i = domain.rfind(".", 0, i)
            suffix = domain[i + 1 :]
            if suffix in self.meta_suffixes:
                # Skip one more part
                i = domain.rfind(".", 0, i)
                continue
            if suffix in self.suffixes:
                continue
            return domain[i + 1 :]

        return domain


def test_domain_shortener() -> None:
    short = DomainShortener()
    assert short("helloworld.fr") == "helloworld.fr"
    assert short("www.helloworld.fr") == "helloworld.fr"
    assert short("alicia.blogspot.com") == "alicia.blogspot.com"
    assert short("hello.alicia.blogspot.com") == "alicia.blogspot.com"
    assert short("peace.gov.uk") == "peace.gov.uk"
    assert short("world.peace.gov.uk") == "peace.gov.uk"
    # I find this one weird but the rule files contains: "*.compute.amazonaws.com"
    assert short("foo.bar.compute.amazonaws.com") == "foo.bar.compute.amazonaws.com"


def domain_shard(domain: str) -> int:
    return cc_net.dedup.str_hash(domain) % 1000


class DomainCollect(jsonql.Transformer):
    def __init__(self, output_file: Path):
        self.stats: Counter[str] = collections.Counter()
        self.output_file = output_file
        self.small_domain: DomainShortener = None

    def _prepare(self):
        self.small_domain = DomainShortener()

    def do(self, document: dict):
        domain = self.small_domain(document["source_domain"])
        self.stats[domain] += 1

    def close(self, failed=False):
        self.log(f"Found {len(self.stats)} unique domains.")
        self.log(f"Most common domains are: {self.stats.most_common(5)}")
        output = self.output_file
        if failed:
            output = self.output_file.with_suffix(".failed.tsv")
        self.log(f"Writing results to {self.output_file} ...")
        domains = sorted(self.stats.items(), key=lambda x: -x[1])
        with jsonql.open_write(output) as o:
            for d, count in domains:
                print(d, count, sep="\t", file=o)
        self.log(f"Done. Results are in {output}")


def group_files_per_lang(
    lang_list: List[str],
    folder: Path,
    pattern: str,
    target_size: int = 40 * 1024**3,
    split: str = "",
):
    for lang in lang_list:
        split = split if lang in ("en", "fr") else ""
        pat = pattern.format(lang=lang, split=split)
        files = [fixup_file(f) for f in folder.glob(pat)]
        if not files:
            log.warning(f"No files found for language: {lang} at {folder}/{pat}")
            continue
        for i, lang_group in enumerate(
            cc_net.regroup.determine_groups(files, target_size=target_size)
        ):
            yield lang, i, lang_group


def collect(
    langs: str,
    out: Path = Path("output"),
    data: Path = Path("data"),
    execution: str = "slurm",
):
    assert data.exists(), f"Data not found at {data}"
    out.mkdir(exist_ok=True)
    out = out.resolve()
    # TODO: move lang_list to group_files_per_lang
    lang_list = all_langs(data) if langs == "all" else langs.split(",")

    groups = []
    outputs = []
    for lang, i, lang_group in group_files_per_lang(
        lang_list, data / "regroup", "*/{lang}_*.json.gz"
    ):
        shard_out = out / f"{lang}_{i:04d}.tsv"
        if shard_out.exists():
            continue
        outputs.append(shard_out)
        groups.append(lang_group)

    executor = cc_net.execution.get_executor(
        "domain_stats",
        out / "logs",
        execution,
        timeout_hour=24,
        mem_gb=4,
        cpus=2,
        task_parallelism=100,
    )

    executor(_collect_stats, groups, outputs)


def _collect_stats(files: List[Path], output_file: Path):
    jsonql.run_pipes(
        DomainCollect(output_file),
        inputs=read_files(files, skip_invalid=True),
    )


def all_langs(data: Path) -> List[str]:
    files = (data / "regroup").glob(f"*/*_0000.json.gz")
    langs = {f.name.split("_")[0] for f in files}
    return sorted(langs)


def read_files(files: list, skip_invalid: bool = True) -> Iterable[dict]:
    # TODO this error handling should be in `open_read`
    json_reader = jsonql.JsonReader()
    for file in files:
        if not file.exists():
            raise FileNotFoundError(file)
        try:
            i = 0
            for line in jsonql.open_read(file):
                doc = json_reader(line)
                if doc is None:
                    continue
                # Check expected keys in case of malformed data
                assert "language" in doc
                assert "digest" in doc
                assert "raw_content" in doc
                assert "source_domain" in doc
                assert "url" in doc
                i += 1
                yield doc
        except EOFError:
            continue
        except (zlib.error, OSError, UnicodeDecodeError) as e:
            if skip_invalid:
                log.error(f"Skipping {file}[{i}:] after exception: {e}")
                continue
            raise


def fixup_file(file: Path) -> Path:
    filename = str(file)
    if "/2019-09/" in filename:
        # /datasets01_101 doesn't exist anymore but we still have symlinks to it
        filename = str(file.resolve())
        return Path(filename.replace("/datasets01_101/", "/datasets01/"))

    return file


class DomainFilter(jsonql.Transformer):
    def __init__(self, domains_file: List[Path]):
        self.domains_file = domains_file
        self.domains: Set[str] = set()
        self.small_domain: DomainShortener = None

    def _prepare(self):
        self.small_domain = DomainShortener()
        for domain_line in jsonql.open_read(self.domains_file):
            self.domains.add(domain_line.split("\t", 1)[0])
        self.log(f"Will keep data from {len(self.domains)} unique domains")

    def do(self, doc: dict) -> Optional[dict]:
        domain = self.small_domain(doc["source_domain"])
        if domain not in self.domains:
            return None
        doc.pop("tokenized", None)
        doc["domain_short"] = domain
        doc["domain_shard"] = domain_shard(domain)
        return doc


def filter(
    domains: str,
    lang: str = "en",
    out: Path = Path("output/sites"),
    data: Path = Path("data"),
    execution: str = "slurm",
):
    lang_list = lang.split(",")
    domains_files = [Path(f) for f in sorted(glob.glob(domains))]
    log.info(f"Received {len(domains_files)} list of domains.")
    assert len(domains_files) > 0
    assert data.exists(), f"Data not found at {data}"
    out.mkdir(exist_ok=True)
    out = out.resolve()

    group_langs = []
    groups: List[List[Path]] = []
    for lang, i, lang_group in group_files_per_lang(
        lang_list, data / "regroup", "*/{lang}_{split}*.json.gz"
    ):
        groups.append(lang_group)
        group_langs.append(lang)

    executor = cc_net.execution.get_executor(
        "domain_filter",
        out / "logs",
        execution,
        timeout_hour=24,
        mem_gb=4,
        cpus=2,
        task_parallelism=100,
    )

    domain_filter = DomainFilter(domains_files)

    def _filter_domains(files: List[Path], lang: str):
        # split per domain names.
        # create intermediary folder to avoid having too many websites
        pattern = str(out / "{domain_shard}" / files[0].name)
        jsonql.run_pipes(
            domain_filter,
            jsonql.split(pattern=pattern, mkdir=True),
            inputs=read_files(files, skip_invalid=True),
        )

    executor(_filter_domains, groups, group_langs)


class MultiWriterCache:
    def __init__(self, max_len: int = 1024):
        self.handles = {}
        self.queue = collections.deque()
        self.max_len = max_len

    def __getitem__(self, file: Path):
        o = self.handles.get(file)
        if o is not None:
            return o

        # We can't use compression efficiently since we close and reopen the files
        file.parent.mkdir(exist_ok=True)
        o = open(file, "at", encoding="utf-8")
        self.handles[file] = o
        self.queue.append(file)

        # Close files to avoid leaking too much resources
        if len(self.queue) > self.max_len:
            oldest_file = self.queue.popleft()
            oldest_handle = self.handles.pop(oldest_file)
            oldest_handle.close()

        return o

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for o in self.handles.values():
            o.close()
        self.queue.clear()
        self.handles = {}


def regroup_sites(sites_dir: Path, output_dir: Path, lang: str = ""):
    json_reader = jsonql.JsonReader()
    if output_dir.exists():
        # since we are appending to files we need to erase previous content
        for file in output_dir.iterdir():
            file.unlink()
    else:
        output_dir.mkdir()
    lang_list = None if not lang else set(lang.split(","))

    def _write_to_lett_file(outfiles: MultiWriterCache, doc: dict) -> None:
        lang = doc["language"]
        if lang_list is not None and lang not in lang_list:
            return

        site = doc["domain_short"]
        out = outfiles[output_dir / (site + ".txt")]
        uri = doc["url"]
        segment = doc.get("cc_segment", "CC_2019_09_is_missing_segment")
        digest = doc["digest"]
        text = base64.b64encode(doc["raw_content"].encode("utf-8")).decode("ascii")
        print(lang, segment, digest, uri, "", text, sep="\t", file=out)
        out.flush()

    with MultiWriterCache() as outfiles:
        for file in sites_dir.iterdir():
            if not file.name.endswith(".json.gz"):
                continue
            if lang_list is not None:
                file_lang = file.name.split("_")[0]
                if file_lang not in lang_list:
                    continue

            for doc in read_files([file], skip_invalid=True):
                _write_to_lett_file(outfiles, doc)

    # Compress each website
    txt_files = [file for file in output_dir.iterdir() if file.suffix == ".txt"]
    log.info(f"Found {len(txt_files)} websites, will start compression ...")

    i = 0
    with multiprocessing.Pool(4) as pool:
        for _ in pool.imap_unordered(_compress_file, txt_files):
            i += 1
            log.info(f"Compressed {i} / {len(txt_files)} files")


def _compress_file(txt_file: Path):
    xz_file = txt_file.with_suffix(".xz")
    with open(txt_file, "rb") as f, lzma.open(xz_file, "wb") as o:
        for line in f:
            o.write(line)
    txt_file.unlink()


def regroup(
    input: Path = Path("output/sites_split"),
    out: Path = Path("output/sites"),
    lang: str = "",
    execution: str = "slurm",
):
    out.mkdir(exist_ok=True)
    out = out.resolve()
    executor = cc_net.execution.get_executor(
        "regroup_sites",
        out / "logs",
        execution,
        timeout_hour=24,
        mem_gb=12,
        cpus=2,
        task_parallelism=100,
    )

    in_dirs = [input / f"{i}" for i in range(1000)]
    out_dirs = [out / f"{i}" for i in range(1000)]

    executor(regroup_sites, in_dirs, out_dirs, itertools.repeat(lang))


def load_domains_stats(domains_file: List[Path]) -> Dict[str, int]:
    domains: Dict[str, int] = {}
    for i, line in enumerate(jsonql.open_read(domains_file)):
        line = line.strip("\n")
        if not line:
            continue
        d, n = line.split("\t", 1)
        domains[d] = domains.get(d, 0) + int(n)
    return domains


def load_english_stats(stats: Path, merged_stats: Path) -> Dict[str, int]:
    if merged_stats.exists():
        return load_domains_stats([merged_stats])

    english_stats = load_domains_stats(list(stats.glob("v1/en_*.tsv")))
    full_en_stats = list(english_stats.items())
    full_en_stats.sort(key=lambda x: x[1], reverse=True)
    with jsonql.open_write(merged_stats) as o:
        for site, n in full_en_stats:
            print(site, n, sep="\t", file=o)
    return english_stats


def promising(stats: Path = Path("output/stats/")):
    out = stats / "intersect"
    out.mkdir(exist_ok=True)
    english_stats = load_english_stats(stats, out / "en.tsv")
    langs = {f.name.split("_")[0] for f in stats.glob(f"v2/*.tsv")}

    for lang in langs:
        target_stats = load_domains_stats(list(stats.glob(f"v2/{lang}_*.tsv")))
        intersection = [
            (s, n1, english_stats[s])
            for s, n1 in target_stats.items()
            if s in english_stats
        ]
        intersection.sort(key=lambda x: min(x[1], x[2]), reverse=True)

        print("most promising sites:")
        print("url", lang, "en", "shard")
        for site, n1, n2 in intersection[:20]:
            print(site, n1, n2, domain_shard(site), sep="\t")

        with jsonql.open_write(out / f"{lang}.tsv") as o:
            for site, n1, n2 in intersection:
                print(site, n1, n2, domain_shard(site), sep="\t", file=o)


def site_info(site: str, out: Path = Path("output")):
    stats = out / "stats/intersect"
    shard = domain_shard(site)
    file = out / f"sites_regroup/{shard}/{site}.xz"
    subprocess.run(["ls", "-lhSr", str(file)])
    subprocess.run(
        f"xzcat {file} | cut -f1 | sort |  uniq -c",
        shell=True,
    )


def read_websites_list(websites: Path) -> Tuple[Set[str], Set[str]]:
    sites = set()
    langs = set()
    shorten = DomainShortener()
    with websites.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            domain, dialect = parts[0], parts[1]
            # if shorten(domain) != domain:
            #     print(f"{shorten(domain)} != {domain}")
            langs.add(dialect)
            sites.add(domain)
    log.info(f"Loaded {len(sites)} sites across {len(langs)} languages")
    return sites, langs


CC200_XL = Path("/large_experiments/nllb/mmt/data/monolingual/cc200xl_v4")


@long_running
def regroup_cc200(
    out: Path = Path("output/sites_cc200_split"),
    websites: Path = Path("/checkpoint/guw/hmine/stats/websites.tsv"),
    execution: str = "slurm",
    resume: str = "54872887",
) -> None:

    out.mkdir(exist_ok=True)
    executor = cc_net.execution.get_executor(
        "regroup_sites",
        out / "logs",
        execution,
        timeout_hour=24,
        mem_gb=10,
        cpus=1,
        task_parallelism=100,
    )

    if not resume:
        sites, langs = read_websites_list(websites)
        lang_files: List[Path] = []
        for lang in sorted(langs):
            lang_files.extend(sorted(CC200_XL.glob(f"cc200xl.{lang}.*.xz")))
        groups = list(cc_net.regroup.determine_groups(lang_files))
        # English files are big
        groups.extend([e] for e in sorted(CC200_XL.glob(f"cc200xl.eng.*.xz")))
        tasks = [WebsiteExtractor(out, sites, group) for group in groups]

    else:
        tasks = []
        for i in range(1_000_000):
            prev_job = submitit.core.utils.JobPaths(out / "logs", f"{resume}_{i}")
            if not prev_job.submitted_pickle.exists():
                log.info(f"Will resume {len(tasks)} jobs from originally {i + 1} jobs")
                break
            progress = _reload(prev_job)
            if progress is None:
                continue
            tasks.append(progress)

    executor(WebsiteExtractor.__call__, tasks)


def _reload(job: submitit.core.utils.JobPaths) -> Optional["WebsiteExtractor"]:

    if job.result_pickle.exists():
        status, res = pickle.loads(job.result_pickle.read_bytes())
        if status == "success":
            return None

    requeued = (
        job.submitted_pickle.stat().st_mtime > job.submission_file.stat().st_mtime
    )
    if requeued:
        breakpoint()
        log.info("Resuming job with progress")
    delayed_fn = pickle.loads(job.submitted_pickle.read_bytes())
    (extractor,) = delayed_fn.args
    return extractor


class WebsiteExtractor(submitit.helpers.Checkpointable):
    def __init__(self, outdir: Path, sites: Set[str], files: List[Path]):
        self.sites = sites
        self.read_lines = -1
        self.files = files
        self.outdir = outdir

    def __call__(self):
        shorten = DomainShortener()

        with MultiWriterCache() as outfiles:
            while self.files:
                file = self.files[0]
                self.read_lines = -1
                lang = file.name.split(".")[1]
                try:
                    with lzma.open(file, "rt", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i < self.read_lines:
                                continue
                            self.read_lines = i
                            line = line.rstrip("\n")
                            text, url, segment, hash, paragraph = line.rsplit("\t", 4)
                            domain = shorten(urllib.parse.urlparse(url).netloc)
                            if domain not in self.sites:
                                continue
                            domain_file = (
                                self.outdir
                                / str(domain_shard(domain))
                                / (domain + ".txt")
                            )
                            print(line, lang, sep="\t", file=outfiles[domain_file])

                except EOFError:
                    pass

                self.files.pop(0)


if __name__ == "__main__":
    func_argparse.main(collect, filter, regroup, promising, site_info, regroup_cc200)

"""Sample commands to inspect results

# Find top sites for lang 'as' in shard 2
zcat output/sites/2/*as_all*.json.gz | jq -r .domain_short | sort | uniq -c | sort -r
      9 banglachotiall.com
      2 aihik.in

# Number of pages for all languages in 'aihik.in' website

xzcat /checkpoint/guw/hmine/sites_regroup_en_as/*/aihik.in.xz | cut -f1 | sort |  uniq -c
xzcat /checkpoint/guw/hmine/sites_regroup_en_as/*/asomiyapratidin.in.xz | cut -f1 | sort |  uniq -c

"""
