from pathlib import Path

import cc_net.execution
import cc_net.websites
import cc_net.jsonql
import logging
import collections
import multiprocessing
import submitit

log = logging.getLogger("websites")

WARC = Path("/checkpoint/guw/hmine/warc")
LETT = Path("/checkpoint/guw/hmine/lett")
FASTTEXT_MODEL = Path(
    "/large_experiments/nllb/mmt/lidruns/lid_models/2022-02-18_ft_model.bin"
)
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
        task_parallelism=515,
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
                skipped += 1
                continue
            kept += 1
            (outdir / dialect).mkdir(exist_ok=True)
            tasks[dialect].append(website)
    log.info(
        f"Found {kept} documents (skipped {skipped} because ratio is too low), across {len(tasks)} languages."
    )

    sites = []
    outdirs = []

    dialects = sorted(tasks.keys())
    for dialect in dialects:
        dialect_outdir = outdir / dialect
        dialect_outdir.mkdir(exist_ok=True)

        for site_group in cc_net.jsonql.grouper(tasks[dialect], 100):
            sites.append(site_group)
            outdirs.append(dialect_outdir)

    log.info(f"Will download {kept} documents across {len(sites)} jobs")
    ex(cc_net.websites.dl_batch, sites, outdirs)


def one(
    website: str,
    outdir: Path = WARC,
    execution: str = "slurm,slurm_partition=learnaccel",
    fasttext: Path = FASTTEXT_MODEL,
    ratio: float = 0.05,
):
    ex = cc_net.execution.get_executor(
        "dl_websites",
        WARC / "logs",
        execution,
        timeout_hour=3 * 24,
        task_parallelism=515,
    )
    lett_file = LETT / (website + ".gz")
    warc_file = outdir / (website + ".gz")
    seq = submitit.helpers.FunctionSequence()
    seq.add(cc_net.websites.dl_from_cc, website, warc_file)
    seq.add(cc_net.websites.extract_lett, warc_file, fasttext, lett_file)
    ex(submitit.helpers.FunctionSequence.__call__, [seq])

def lett(
    langs: str = "tir",
    warc: Path = WARC,
    lett: Path = LETT,
    fasttext: Path = FASTTEXT_MODEL,
    execution: str = "local",
):
    _langs = langs.split(",")

    n_files = 0
    stats = collections.defaultdict(int)
    tasks = []
    for lang in _langs:
        (lett / lang).mkdir(exist_ok=True)
        for file in (WARC / lang).iterdir():
            if ".tmp." in file.name:
                continue
            if file.suffix == ".progress":
                continue
            if file.is_dir():
                continue
            if (file.parent / (file.name + ".gz")).exists():
                continue

            lett_file_name = file.name
            if not lett_file_name.endswith(".gz"):
                lett_file_name += ".gz"
            lett_file = lett / lang / lett_file_name
            if lett_file.exists():
                continue
            tasks.append((file, fasttext, lett_file))

    print(f"Stats on {n_files} files: {stats}")

    batches = [submitit.helpers.FunctionSequence()]
    batch_size = len(tasks) / 200
    for task in tasks:
        if len(batches[-1]) > batch_size:
            batches.append(submitit.helpers.FunctionSequence())
        batches[-1].add(cc_net.websites.extract_lett, *task)

    print(f"Grouped {len(tasks)} in {len(batches)} groups")

    ex = cc_net.execution.get_executor(
        "dl_websites",
        WARC / "logs",
        execution,
        timeout_hour=3*24,
        task_parallelism=200,
    )

    ex(submitit.helpers.FunctionSequence.__call__, batches)

    # Aggregate stats
    with multiprocessing.Pool(16) as pool:
        for file_stats in pool.starmap(cc_net.websites.extract_lett, tasks):
            for k, v in file_stats.items():
                n_files += 1
                stats[k] += v
    print(f"Stats on {n_files} files: {stats}")

    # for task in tasks:
    #     cc_net.websites.extract_lett(*task)
    #     print(f"Converted {task}")


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(dl, lett, one)
