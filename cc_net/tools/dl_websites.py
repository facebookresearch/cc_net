from pathlib import Path

import cc_net.execution
import cc_net.websites
import cc_net.jsonql
import logging
import collections

log = logging.getLogger("websites")

WARC = Path("/checkpoint/guw/hmine/warc")
WEBSITES = Path("/checkpoint/guw/hmine/stats/websites.tsv")
RATIO_PAGES_COL = 5


def dl(
    websites: Path = WEBSITES,
    outdir: Path = WARC,
    execution: str = "slurm,slurm_partition=learnaccel",
    ratio: float = 0.05,
):
    ex = cc_net.execution.get_executor(
        "dl_websites",
        WARC / "logs",
        execution,
        timeout_hour=3 * 24,
        task_parallelism=250,
    )

    tasks = collections.defaultdict(list)
    kept, skipped = 0, 0
    with open(websites, "r") as f:
        header = f.readline().split("\t")
        assert header[2 + RATIO_PAGES_COL] == "ratio_pages"
        for line in f:
            website, dialect, raw_counters = line.rstrip().split("\t", 2)
            counters = [float(i) for i in raw_counters.split("\t")]
            ratio_pages = counters[RATIO_PAGES_COL]
            if ratio_pages < ratio:
                # log.warn(
                #     f"Skipping website {website} in {dialect} because ratio is {ratio_pages}"
                # )
                skipped+= 1
                continue
            kept += 1
            (outdir / dialect).mkdir(exist_ok=True)
            tasks[dialect].append(website)
    log.info(f"Found {kept} documents (skipped {skipped} because ratio is too low), across {len(tasks)} languages.")

    sites = []
    outdirs = []

    for dialect in tasks:
        dialect_outdir = outdir / dialect
        dialect_outdir.mkdir(exist_ok=True)

        for site_group in cc_net.jsonql.grouper(tasks[dialect], 100):
            sites.append(site_group)
            outdirs.append(dialect_outdir)

    log.info(f"Will download {kept} documents across {len(sites)} jobs")
    ex(cc_net.websites.dl_batch, sites, outdirs)


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(dl, warc2text)

