# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import logging
import multiprocessing
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sized

from typing_extensions import Protocol


class Executor(Protocol):
    def __call__(self, function: Callable[..., str], *args: Iterable) -> None:
        ...


def get_executor(
    name: str,
    log_dir: Path,
    execution: str,
    timeout_hour: float = 1.0,
    mem_gb: int = 1,
    cpus: int = 1,
    task_parallelism: int = -1,
    options: dict = {},
) -> Executor:

    execution_mode = execution.split(",")[0]
    options.update(
        {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in execution.split(",")[1:]}
    )
    if execution_mode == "slurm":
        ex = get_submitit_executor(
            name, log_dir, timeout_hour, mem_gb, cpus, task_parallelism, options
        )
        if ex is not None:
            return ex

    if execution_mode == "mp":
        return MpExecutor(log_dir, cpus, task_parallelism)

    return debug_executor


def get_submitit_executor(
    name: str,
    log_dir: Path,
    timeout_hour: float,
    mem_gb: int,
    cpus: int,
    task_parallelism: int,
    options: dict,
) -> Optional[Executor]:
    try:
        import submitit

        ex = submitit.AutoExecutor(log_dir)
    except ImportError:
        warnings.warn(f"Failed to import submitit, will try another executor.")
        return None
    except RuntimeError as e:
        warnings.warn(
            f"Failed to create submitit.AutoExecutor, will try another executor. ({e})"
        )
        return None

    class SubmititRetryOnTimeout(submitit.helpers.Checkpointable):
        def __init__(self, fn: Callable):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    ex.update_parameters(
        name=name,
        timeout_min=int(timeout_hour * 60),
        mem_gb=mem_gb,
        cpus_per_task=cpus,
        slurm_array_parallelism=task_parallelism,
        **options,
    )

    def submit_and_wait(function: Callable[..., str], *args: Iterable):
        f_name = function.__name__

        assert len(args) > 0, f"No arguments passed to {f_name}"
        approx_length = _approx_length(*args)

        print(f"Submitting {f_name} in a job array ({approx_length} jobs)")
        jobs = ex.map_array(function, *args)
        if not jobs:
            return
        failed_jobs = []
        done = 0
        total = len(jobs)
        job_array_id = jobs[0].job_id.split("_")[0]
        print(f"Started {f_name} in job array {job_array_id} ({len(jobs)} jobs).")
        for job in submitit.helpers.as_completed(jobs):
            done += 1
            e = job.exception()
            if not e:
                print(f"Finished job {job.job_id} ({done} / {total}).", job.result())
                continue

            print(f"Failed job {job.job_id} ({done} / {total}):", e)
            failed_jobs.append(job)

        if failed_jobs:
            n_failures = 10
            message = f"{len(failed_jobs)} / {done} jobs failed while running {f_name}"
            print(message)
            for job in failed_jobs[:n_failures]:
                print(f"Failed {job.job_id} -> {job.paths.stderr}")
            if len(failed_jobs) > n_failures:
                print(f"... ({len(failed_jobs) - n_failures} failed job skipped)")
            raise Exception(message)

    return submit_and_wait


def debug_executor(function: Callable[..., Optional[str]], *args: Iterable) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    approx_length = _approx_length(*args)
    for i, x in enumerate(zip(*args)):
        try:
            message = function(*x)
        except Exception:
            try:
                import ipdb as pdb  # type: ignore
            except ImportError:
                import pdb  # type: ignore
            import traceback

            traceback.print_exc()
            print("")
            pdb.post_mortem()
            sys.exit(1)
        if message is not None:
            print(message, f"({i + 1} / {approx_length})")


def _approx_length(*args: Iterable):
    for a in args:
        if isinstance(a, Sized):
            return len(a)
    return -1


GLOBAL_FUNCTIONS: Dict[str, Callable[..., Optional[str]]] = {}


def global_fn(args) -> Optional[str]:
    f_name = args[0]
    f = GLOBAL_FUNCTIONS[f_name]
    return f(*args[1:])


class MpExecutor(Executor):
    def __init__(self, log_dir: Path, cpus: int, task_parallelism: int):
        self.log_dir = log_dir
        if task_parallelism < 0:
            task_parallelism = os.cpu_count() or 1
        self.processes = min(task_parallelism // cpus, os.cpu_count())

    def __call__(self, function: Callable[..., Optional[str]], *args: Iterable):

        f_name = function.__name__
        global GLOBAL_FUNCTIONS
        if f_name in GLOBAL_FUNCTIONS:
            assert (
                function == GLOBAL_FUNCTIONS[f_name]
            ), f"Conflicting name between {function} and {GLOBAL_FUNCTIONS[f_name]}"
        else:
            GLOBAL_FUNCTIONS[f_name] = function

        approx_length = _approx_length(*args)

        print(
            f"Starting {f_name} over {self.processes} processes ({approx_length} tasks)."
        )
        with multiprocessing.Pool(processes=self.processes) as pool:
            i = 0
            for message in pool.imap_unordered(
                global_fn, zip(itertools.repeat(f_name), *args)
            ):
                i += 1
                print(message, f"({i} / {approx_length})")
