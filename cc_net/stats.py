import base64
import collections
import itertools
import glob
import logging
import zlib
from typing import Counter, List, Iterable, Optional, Set
from pathlib import Path

import func_argparse

import cc_net.dedup
import cc_net.regroup
import cc_net.execution
from cc_net import jsonql

log = logging.getLogger("stats")

PUBLIC_SUFFIX_LIST = Path(__file__).parent / "data" / "public_suffix_list.dat"


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


def group_files_per_lang(lang_list: List[str], folder: Path, pattern: str, target_size: int = 40 * 1024 ** 3):
    for lang in lang_list:
        files = [fixup_file(f) for f in folder.glob(pattern.format(lang=lang))]
        if not files:
            log.warning(f"No files found for language: {lang} at {folder}/{pattern.format(lang=lang)}")
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
    for lang, i, lang_group in group_files_per_lang(lang_list, data / "regroup", "*/{lang}_*.json.gz"):
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
                assert "url" in doc
                assert "segment" in doc
                assert "language" in doc
                assert "domain_short" in doc
                i += 1
                yield doc
        except EOFError:
            continue
        except (zlib.error, OSError, UnicodeDecodeError, AssertionError) as e:
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
        doc["domain_shard"] = cc_net.dedup.str_hash(domain) % 1000
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
    for lang, i, lang_group in group_files_per_lang(lang_list, data / "regroup", "*/{lang}_*.json.gz"):
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


def regroup_sites(sites_dir: Path, output_dir: Path, lang: str = ""):
    json_reader = jsonql.JsonReader()
    output_dir.mkdir(exist_ok=True)
    outfiles = {}
    lang_list = None if not lang else set(lang.split(","))

    def _write_to_lett_file(doc: dict) -> None:
        lang = doc["language"]
        if lang_list is not None and lang not in lang_list:
            return

        site = doc["domain_short"]
        site_file = output_dir / (site + ".xz")
        if site_file not in outfiles:
            outfiles[site_file] = jsonql.open_write(site_file).__enter__()
        out = outfiles[site_file]
        uri = doc["url"]
        segment = doc.get("cc_segment", "CC_2019_09_is_missing_segment")
        digest = doc["digest"]
        text = base64.b64encode(doc["raw_content"].encode("utf-8")).decode("ascii")
        print(lang, segment, digest, uri, "", text, sep="\t", file=out)

    try:
        for doc in read_files(list(sites_dir.glob("*.json.gz")), skip_invalid=True):
            _write_to_lett_file(doc)
    finally:
        for o in outfiles.values():
            o.close()


def regroup(
    input: Path = Path("output/sites"),
    out: Path = Path("output/sites_regroup"),
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
        mem_gb=4,
        cpus=2,
        task_parallelism=100,
    )

    in_dirs = [input / f"{i}" for i in range(1000)]
    out_dirs = [out / f"{i}" for i in range(1000)]

    executor(regroup_sites, in_dirs, out_dirs, itertools.repeat(lang))


if __name__ == "__main__":
    func_argparse.main(collect, filter, regroup)

"""Sample commands to inspect results

# Find top sites for lang 'as' in shard 2
zcat output/sites/2/*as_all*.json.gz | jq -r .domain_short | sort | uniq -c | sort -r
      9 banglachotiall.com
      2 aihik.in

# Number of pages for all languages in 'aihik.in' website

xzcat /checkpoint/guw/hmine/sites_regroup_en_as/*/aihik.in.xz | cut -f1 | sort |  uniq -c
xzcat /checkpoint/guw/hmine/sites_regroup_en_as/*/asomiyapratidin.in.xz | cut -f1 | sort |  uniq -c

"""
