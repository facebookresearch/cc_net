import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from cc_net import jsonql, minify, process_wet_file
from cc_net.execution import get_executor


class Transposer(jsonql.Transformer):
    """Reads a shard created by mine.py and split the information in segments."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.segments: Dict[str, List[dict]] = {}

    def _prepare(self) -> None:
        self.output_dir.mkdir(exist_ok=True)

    def do(self, doc: dict) -> None:
        doc_seg = doc["cc_segment"]
        if doc_seg not in self.segments:
            self.segments[doc_seg] = []
        self.segments[doc_seg].append(doc)
        return None

    def speed_summary(self) -> str:
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.processed / delay
        return f"Processed {self.processed:_} documents across {len(self.segments)} segments in {h:.2}h ({s:5.1f} doc/s)."

    def close(self):
        """Dump metadata by segments"""
        for segment, meta in self.segments.items():
            file_name = segment.split("/")[-1]
            assert file_name.endswith(".warc.wet.gz")
            file_name = file_name.replace(".warc.wet.gz", ".json.gz")
            out = self.output_dir / file_name

            # this is not sorted, we should sort it afterward
            with jsonql.smart_open(out, "w") as o:
                for m in meta:
                    print(json.dumps(m), file=o)


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


def _transpose_files(files: List[Path], output_dir: Path):
    mi = minify.Minifier()
    tr = Transposer(output_dir)
    return jsonql.run_pipes(mi, tr, file=files, output="/dev/null")


def transpose(
    dump: str, parallelism: int = 100, execution: str = "slurm", filter: str = None
) -> None:

    data = Path("data") / "regroup" / dump
    assert data.exists(), f"Dump directory not found: {data}"
    output = Path("data") / "transposed" / dump
    output.mkdir(exist_ok=True, parents=True)

    files = [f for f in data.iterdir() if f.name.endswith(".json.gz")]
    if filter is not None:
        files = [f for f in files if re.match(filter, f.name)]

    print(f"Found {len(files)} files in {data}")
    groups = list(jsonql.grouper(files, len(files) // parallelism))

    ex = get_executor(
        "transpose",
        output / "logs",
        execution,
        timeout_hour=10,
        mem_gb=32,
        cpus=2,
        task_parallelism=parallelism,
    )

    tmp_dirs = [output / f"{i}" for i in range(len(groups))]
    ex(_transpose_files, groups, tmp_dirs)


class LinearUnminifier(minify.Unminifier):
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder
        self.segment: str = ""
        self.segments_read_twice = 0

    def fetch_metadata(self, segment: str) -> None:
        file_name = segment.split("/")[-1]
        assert file_name.endswith(".warc.wet.gz")
        file_name = file_name.replace(".warc.wet.gz", ".json.gz")
        with jsonql.smart_open(self.folder / file_name) as o:
            metadata = jsonql.read_jsons(o)
            self.metadata = {m["digest"]: m for m in metadata}

        self.segment = segment
        if segment in self._segments:
            self.log("Cache miss")
            self.segments_read_twice += 1
        self._segments.add(segment)

    def do(self, doc: dict) -> Optional[dict]:
        if self.segment != doc["cc_segment"]:
            self.fetch_metadata(doc["cc_segment"])
        return super().do(doc)


def unminify(
    folder: Path, dump: str, output_dir: Path, shard: int = 0, cache_dir: Path = None
):
    unminifier = LinearUnminifier(folder)
    output = output_dir / dump / f"all_{shard:04d}.json.gz"
    tmp = jsonql._tmp(output)
    cc = process_wet_file.CCShardReader(dump, shard, 1600, cache_dir=cache_dir)

    jsonql.run_pipes(unminifier, file=cc, output=tmp)
    tmp.rename(output)
    f_size = sum(Path(f).stat().st_size for f in folder.glob("*.gz"))
    o_size = output.stat().st_size
    mb = 1024 ** 2
    return f"Unminified {output} ({f_size // mb:_}Mb -> {o_size // mb:_}Mb)"


if __name__ == "__main__":
    import func_argparse

    func_argparse.main(transpose, unminify)
