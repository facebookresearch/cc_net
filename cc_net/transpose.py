import collections
import json
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

from cc_net import jsonql, mine, minify, process_wet_file
from cc_net.execution import get_executor


class Metadata(NamedTuple):
    url: str
    digest: str
    cc_segment: str
    language: str
    language_score: float
    perplexity: float
    bucket: str
    hashes: str


class Transposer(jsonql.Transformer):
    """Reads a shard created by mine.py and split the information in segments."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.segments: Dict[str, List[Metadata]] = collections.defaultdict(list)

    def do(self, doc: dict) -> None:
        metadata = Metadata(
            doc["url"],
            doc["digest"],
            doc["cc_segment"],
            doc["language"],
            doc["language_score"],
            doc.get("perplexity", -1),
            doc["bucket"],
            doc["hashes"],
        )
        self.segments[metadata.cc_segment].append(metadata)
        return None

    def speed_summary(self) -> str:
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.processed / delay
        return f"Processed {self.processed:_} documents across {len(self.segments)} segments in {h:.2}h ({s:5.1f} doc/s)."

    def close(self):
        """Dump metadata by segments"""
        tmp = self.output_dir.with_suffix(".tmp")
        tmp.mkdir(exist_ok=True)
        for segment, meta in self.segments.items():
            file_name = segment.split("/")[-1]
            assert file_name.endswith(".warc.wet.gz")
            file_name = file_name.replace(".warc.wet.gz", ".json.gz")
            out = tmp / file_name

            # this is not sorted, we should sort it afterward
            with jsonql.smart_open(out, "w") as o:
                for m in meta:
                    d = m._asdict()
                    if m.perplexity < 0:
                        d.pop("perplexity")
                    print(json.dumps(d), file=o)
        tmp.rename(self.output_dir)


def sort_metadata(metadata_file: Path, cache_dir: Path = None) -> None:
    # Idempotent
    with jsonql.smart_open(metadata_file) as lines:
        metadata = {m["digest"]: m for m in jsonql.read_jsons(lines)}

    segment: str = ""
    for d in metadata:
        segment = metadata[d]["cc_segment"]
        break
    tmp = jsonql._tmp(metadata_file)
    with open(tmp, "w") as o:
        for doc in process_wet_file.CCSegmentsReader([segment], cache_dir=cache_dir):
            m = metadata.pop(doc["digest"])
            print(m, file=o)

    assert len(metadata) == 0
    tmp.rename(metadata_file)


def _transpose_file(file: Path, output_dir: Path):
    mi = minify.Minifier()
    tr = Transposer(output_dir)
    return jsonql.run_pipes(mi, tr, file=file, output="/dev/null")


def transpose(
    files: List[Path],
    output_dir: Path,
    parallelism: int = 200,
    execution: str = "slurm",
) -> None:
    files = minify._expand_files(files)
    output_dir.mkdir(exist_ok=True)
    ex = get_executor(
        "transpose",
        output_dir / "logs",
        execution,
        timeout_hour=10,
        mem_gb=64,
        cpus=2,
        task_parallelism=parallelism,
    )

    tr_dirs = [output_dir / f.stem.replace(".json", "") for f in files]
    # skip existing directories
    files = [f for (f, o) in zip(files, tr_dirs) if not o.exists()]
    tr_dirs = [o for o in tr_dirs if not o.exists()]
    ex(_transpose_file, files, tr_dirs)


def regroup_tr(
    input_dir: Path,
    output_dir: Path,
    dump: str = None,
    config: str = "base",
    execution: str = "slurm",
    parallelism: int = 200,
):
    output_dir.mkdir(exist_ok=True)
    if dump is None:
        dump = output_dir.name

    def _regroup(segment: str) -> None:
        s = segment.split("/")[-1].replace(".warc.wet.gz", ".json.gz")
        parts = [f / s for f in input_dir.iterdir() if (f / s).exists()]
        if not parts:
            print(f"Segment {s} not found at {input_dir}/*/{s}")
            return
        jsonql.run_pipes(file=parts, output=output_dir / s)

    ex = get_executor(
        "regroup_tr",
        output_dir / "logs",
        execution,
        timeout_hour=0.5,
        mem_gb=2,
        cpus=2,
        task_parallelism=parallelism,
    )

    conf = mine.PREDEF_CONFIGS[config]
    if conf.num_segments_per_shard > 0:
        segments = [
            seg for i in range(conf.num_shards) for seg in conf.get_cc_shard(i).segments
        ]
    else:
        segments = process_wet_file.cc_segments(dump)
    ex(_regroup, segments)


class LinearUnminifier(minify.Unminifier):
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder
        self.segment: str = ""
        self.segments_read_twice = 0

    def fetch_metadata(self, segment: str) -> None:
        file_name = segment.split("/")[-1]
        assert file_name.endswith(".warc.wet.gz")
        meta_file = self.folder / file_name.replace(".warc.wet.gz", ".json.gz")
        assert (
            meta_file.exists()
        ), f"Couldn't found metadata file for segment {segment} at {file_name}"
        k = minify.get_doc_key
        with jsonql.smart_open(meta_file) as o:
            self.metadata = {k(m["digest"]): m for m in jsonql.read_jsons(o)}

        self.segment = segment
        if segment in self._segments:
            self.log("Cache miss")
            self.segments_read_twice += 1
        self._segments.add(segment)

    def do(self, doc: dict) -> Optional[dict]:
        if self.segment != doc["cc_segment"]:
            self.fetch_metadata(doc["cc_segment"])
        return super().do(doc)


def unminify_shard(
    folder: Path,
    output: Path,
    dump: str,
    shard: int,
    shard_config: mine.Config,
    cache_dir: Path = None,
):
    unminifier = LinearUnminifier(folder)
    tmp = jsonql._tmp(output)
    cc = process_wet_file.CCShardReader(
        dump,
        shard,
        num_shards=shard_config.num_shards,
        num_segments_per_shard=shard_config.num_segments_per_shard,
        cache_dir=cache_dir,
    )

    jsonql.run_pipes(unminifier, file=cc, output=tmp)
    tmp.rename(output)
    f_size = sum(Path(f).stat().st_size for f in folder.glob("*.gz"))
    o_size = output.stat().st_size
    mb = 1024 ** 2
    return f"Unminified {output} ({f_size // mb:_}Mb -> {o_size // mb:_}Mb)"


def unminify(
    folder: Path,
    output_dir: Path,
    dump: str = None,
    config: str = "base",
    execution: str = "slurm",
    parallelism: int = 200,
):
    ex = get_executor(
        "unminify_tr",
        output_dir / "logs",
        execution,
        timeout_hour=0.5,
        mem_gb=2,
        cpus=2,
        task_parallelism=parallelism,
    )
    conf = mine.PREDEF_CONFIGS[config]
    print(conf)
    output_dir.mkdir(exist_ok=True)
    dump_id = output_dir.name if dump is None else dump

    def _unminify_shard(o: Path, shard: int):
        return unminify_shard(folder, o, shard=shard, dump=dump_id, shard_config=conf)

    shards = range(conf.num_shards)
    outs = [output_dir / f"all_{shard:04d}.json.gz" for shard in shards]
    ex(_unminify_shard, outs, shards)


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(transpose, unminify, regroup_tr)
