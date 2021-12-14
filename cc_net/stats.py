import collections
import itertools
import logging
import zlib
from typing import Counter, List, Iterable, Set
from pathlib import Path

import func_argparse

import cc_net.regroup
import cc_net.execution
from cc_net import jsonql

log = logging.getLogger("stats")

PUBLIC_SUFFIX_LIST = Path(__file__).parent / "data" / "public_suffix_list.dat"


class DomainStats(jsonql.Transformer):
    def __init__(self, output_file: Path):
        self.stats: Counter[str] = collections.Counter()
        self.output_file = output_file
        # List of public suffixes that don't indicate a real domain name
        # eg: "blogpost.com", ".gov.uk", ".co.in", ...
        self.suffixes: Set[str] = set()
        self.meta_suffixes: Set[str] = set()

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

    def small_domain(self, domain):
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


def test_small_domains() -> None:
    d = DomainStats(Path("/tmp") / __name__)
    d._prepare()
    small = d.small_domain

    assert small("helloworld.fr") == "helloworld.fr"
    assert small("www.helloworld.fr") == "helloworld.fr"
    assert small("alicia.blogspot.com") == "alicia.blogspot.com"
    assert small("hello.alicia.blogspot.com") == "alicia.blogspot.com"
    assert small("peace.gov.uk") == "peace.gov.uk"
    assert small("world.peace.gov.uk") == "peace.gov.uk"
    # I find this one weird but the rule files contains: "*.compute.amazonaws.com"
    assert small("foo.bar.compute.amazonaws.com") == "foo.bar.compute.amazonaws.com"


def _get_stats(files: List[Path], output_file: Path):
    jsonql.run_pipes(
        DomainStats(output_file),
        inputs=read_files(files, skip_invalid=True),
    )


def read_files(files: list, skip_invalid: bool = True) -> Iterable[dict]:
    json_reader = jsonql.JsonReader()
    for file in files:
        if not file.exists():
            if skip_invalid:
                log.error(f"Skipping non existing {file}")
                continue
            raise FileNotFoundError(file)
        try:
            i = 0
            for line in jsonql.open_read(file):
                yield json_reader(line)
                i += 1
        except (zlib.error, OSError, UnicodeDecodeError) as e:
            if skip_invalid:
                log.error(f"Skipping {file}[{i}:] after exception: {e}")
                continue
            raise


def all_langs(data: Path) -> List[str]:
    files = (data / "regroup").glob(f"*/*_0000.json.gz")
    langs = {f.name.split("_")[0] for f in files}
    return sorted(langs)


def fixup_file(file: Path) -> Path:
    filename = str(file)
    if "/2019-09/" in filename:
        # /datasets01_101 doesn't exist anymore but we still have symlinks to it
        filename = str(file.resolve())
        return Path(filename.replace("/datasets01_101/", "/datasets01/"))

    return file


def main(
    langs: str,
    out: Path = Path("output"),
    data: Path = Path("data"),
    execution: str = "slurm",
):
    assert data.exists(), f"Data not found at {data}"
    out.mkdir(exist_ok=True)
    out = out.resolve()
    lang_list = all_langs(data) if langs == "all" else langs.split(",")

    groups = []
    outputs = []
    for lang in lang_list:
        files = [fixup_file(f) for f in (data / "regroup").glob(f"*/{lang}_*.json.gz")]
        if not files:
            log.warn(f"No files found for language: {lang}")
        for i, lang_group in enumerate(
            cc_net.regroup.determine_groups(files, target_size=40 * 1024 ** 3)
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

    executor(_get_stats, groups, outputs)


if __name__ == "__main__":
    func_argparse.single_main(main)
