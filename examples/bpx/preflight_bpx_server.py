"""Fail-fast validation for the BPX OpenPI training server."""

import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

import jax
import tyro


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _require_path(path: pathlib.Path, *, directory: bool = False) -> None:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"Required {kind} is missing: {path}")


def main(
    repo_root: pathlib.Path = pathlib.Path("."),
    *,
    min_free_gib: float = 500.0,
    fsdp_devices: int | None = None,
    batch_size: int = 64,
) -> None:
    """Validate software, GPU, disk, base checkpoint, and cleaned data prerequisites."""
    repo_root = repo_root.resolve()
    if platform.system() != "Linux":
        raise RuntimeError(f"BPX training is supported on Linux, got {platform.system()}.")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Expected Python 3.11, got {platform.python_version()}.")

    _require_path(repo_root / "pyproject.toml")
    _require_path(repo_root / ".gitmodules")
    for executable in ("git", "ffmpeg", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is not available on PATH: {executable}")

    submodule_status = _run(["git", "-C", str(repo_root), "submodule", "status", "--recursive"])
    incomplete_submodules = [line for line in submodule_status.splitlines() if line.startswith(("-", "+", "U"))]
    if incomplete_submodules:
        raise RuntimeError(
            "Git submodules are missing or inconsistent; run 'git submodule update --init --recursive'.\n"
            + "\n".join(incomplete_submodules)
        )

    base_root = repo_root / "checkpoints" / "pretrained" / "pi05_base"
    _require_path(base_root, directory=True)
    _require_path(base_root / "assets", directory=True)
    _require_path(base_root / "params", directory=True)
    _require_path(base_root / "params" / "_CHECKPOINT_METADATA")
    _require_path(base_root / "params" / "manifest.ocdbt")

    data_root = repo_root / "datasets" / "cleaned_bpx"
    _require_path(data_root, directory=True)
    _require_path(data_root / "cleaning_audit.csv")
    _require_path(data_root / "cleaning_summary.json")
    records_root = data_root / ".cleaning_records"
    _require_path(records_root, directory=True)
    episode_files = sorted(data_root.glob("episode_*.hdf5"))
    record_files = sorted(records_root.glob("episode_*.json"))
    if len(episode_files) != 59 or len(record_files) != 59:
        raise RuntimeError(
            f"Expected 59 episodes and 59 cleaning records, got {len(episode_files)} and {len(record_files)}."
        )
    summary = json.loads((data_root / "cleaning_summary.json").read_text(encoding="utf-8"))
    expected_summary = {
        "episode_count": 59,
        "frame_count": 67_455,
        "repair_count": 439,
        "rejected_candidate_count": 0,
    }
    mismatches = {
        key: (summary.get(key), expected) for key, expected in expected_summary.items() if summary.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Cleaning summary does not match the audited dataset: {mismatches}")

    disk = shutil.disk_usage(repo_root)
    free_gib = disk.free / 1024**3
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB is free on the repository filesystem; "
            f"at least {min_free_gib:.1f} GiB is required before conversion and checkpointing."
        )

    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu" or not devices:
        raise RuntimeError(f"JAX CUDA backend is unavailable: backend={backend!r}, devices={devices!r}")
    gpu_inventory = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    device_count = jax.device_count()
    fsdp_candidates = [count for count in range(1, device_count + 1) if device_count % count == 0]
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if batch_size % device_count != 0:
        raise RuntimeError(
            f"The requested global batch {batch_size} is not divisible by JAX device count {device_count}; "
            "restrict CUDA_VISIBLE_DEVICES or revise the reviewed batch size."
        )
    if fsdp_devices is not None and fsdp_devices not in fsdp_candidates:
        raise RuntimeError(
            f"Requested FSDP device count {fsdp_devices} is invalid for {device_count} JAX devices; "
            f"choose one of {fsdp_candidates}."
        )

    print(f"PASS: repo_root={repo_root}")
    print(f"PASS: Python={platform.python_version()}, JAX backend={backend}, JAX devices={device_count}")
    print(f"PASS: global_batch={batch_size} is divisible by JAX devices={device_count}")
    print(f"PASS: free_disk={free_gib:.1f} GiB (required >= {min_free_gib:.1f} GiB)")
    print("NVIDIA inventory (index, name, MiB):")
    print(gpu_inventory)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"Valid --fsdp-devices divisors for this JAX device count: {fsdp_candidates}")
    if fsdp_devices is not None:
        print(f"PASS: requested --fsdp-devices={fsdp_devices}")
    print("PASS: complete pi05_base and audited BPX dataset (59 episodes / 67,455 frames)")


if __name__ == "__main__":
    tyro.cli(main)
