from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import pathlib
import re
import sys

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CUDA_DEVICES = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


@dataclasses.dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: pathlib.Path
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    total_units: int | None = None

    @property
    def display(self) -> str:
        return " ".join(_quote(part) for part in self.argv)


def _quote(value: str) -> str:
    if value and all(character.isalnum() or character in "-._/:=," for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _existing_directory(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label}不存在或不是目录: {path}")
    return path


def _output_path(value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser().resolve()
    if path == path.parent:
        raise ValueError(f"{label}不能是文件系统根目录")
    return path


def _name(value: str, label: str) -> str:
    clean = value.strip()
    if not _NAME.fullmatch(clean):
        raise ValueError(f"{label}只能包含字母、数字、点、下划线和连字符")
    return clean


def _positive(value: int, label: str, *, maximum: int | None = None) -> int:
    if value < 1 or (maximum is not None and value > maximum):
        suffix = f"且不能超过 {maximum}" if maximum is not None else ""
        raise ValueError(f"{label}必须为正整数{suffix}")
    return value


def _openpi_root(value: str) -> pathlib.Path:
    root = _existing_directory(value, "OpenPI 根目录")
    for relative in ("scripts/compute_norm_stats.py", "scripts/train.py", "pyproject.toml"):
        if not (root / relative).is_file():
            raise ValueError(f"OpenPI 根目录缺少 {relative}: {root}")
    return root


def _data_environment(values: Mapping[str, str]) -> dict[str, str]:
    names = {
        "hf_lerobot_home": "HF_LEROBOT_HOME",
        "hf_home": "HF_HOME",
        "hf_datasets_cache": "HF_DATASETS_CACHE",
        "tmpdir": "TMPDIR",
        "uv_cache_dir": "UV_CACHE_DIR",
    }
    result = {}
    for field, environment_name in names.items():
        raw = values.get(field, "").strip()
        if raw:
            result[environment_name] = str(_output_path(raw, environment_name))
    return result


def build_audit(project_root: pathlib.Path, values: Mapping[str, object]) -> CommandSpec:
    dataset = _existing_directory(str(values["dataset_root"]), "数据集目录")
    workers = _positive(int(values.get("workers", 32)), "workers", maximum=256)
    report = _output_path(str(values["report_path"]), "审计报告")
    video_check = str(values.get("video_check", "headers"))
    if video_check not in {"none", "headers", "full"}:
        raise ValueError("视频检查只能是 none、headers 或 full")
    return CommandSpec(
        argv=(
            sys.executable,
            "-m",
            "openpi.bpx_studio.audit",
            str(dataset),
            "--workers",
            str(workers),
            "--video-check",
            video_check,
            "--report",
            str(report),
        ),
        cwd=project_root,
    )


def build_spike_job(project_root: pathlib.Path, values: Mapping[str, object], *, apply: bool) -> CommandSpec:
    source = _existing_directory(str(values["source_root"]), "源数据集目录")
    thresholds = pathlib.Path(str(values["thresholds_path"])).expanduser().resolve()
    if not thresholds.is_file():
        raise ValueError(f"毛刺阈值文件不存在: {thresholds}")
    workers = _positive(int(values.get("workers", 32)), "workers", maximum=256)
    report = _output_path(str(values["report_path"]), "毛刺报告")
    argv = [
        sys.executable,
        "-m",
        "openpi.bpx_studio.clean_spikes",
        str(source),
        "--thresholds",
        str(thresholds),
        "--workers",
        str(workers),
        "--report",
        str(report),
    ]
    if apply:
        output = _output_path(str(values["output_root"]), "清洗输出目录")
        if output == source:
            raise ValueError("清洗输出目录不能与源数据集相同")
        argv.extend(("--apply", "--output-root", str(output)))
    return CommandSpec(argv=tuple(argv), cwd=project_root)


def build_hdf5_export(project_root: pathlib.Path, values: Mapping[str, object]) -> CommandSpec:
    source = _existing_directory(str(values["source_root"]), "LeRobot 源数据集")
    output = _output_path(str(values["output_root"]), "HDF5 输出目录")
    if source == output:
        raise ValueError("HDF5 输出目录不能覆盖 LeRobot 权威数据集")
    workers = _positive(int(values.get("workers", 4)), "workers", maximum=64)
    compression = str(values.get("compression", "gzip"))
    if compression not in {"gzip", "lzf", "none"}:
        raise ValueError("HDF5 压缩只能是 gzip、lzf 或 none")
    level = int(values.get("compression_level", 4))
    if not 0 <= level <= 9:
        raise ValueError("gzip 压缩级别必须在 0 到 9 之间")
    return CommandSpec(
        argv=(
            sys.executable,
            "-m",
            "openpi.bpx_studio.hdf5_exporter",
            str(source),
            str(output),
            "--workers",
            str(workers),
            "--compression",
            compression,
            "--compression-level",
            str(level),
        ),
        cwd=project_root,
    )


def build_normalize(values: Mapping[str, object]) -> CommandSpec:
    root = _openpi_root(str(values["openpi_root"]))
    config = _name(str(values["config_name"]), "训练配置名")
    workers = _positive(int(values.get("workers", 32)), "workers", maximum=256)
    env = _data_environment({key: str(value) for key, value in values.items()})
    return CommandSpec(
        argv=("uv", "run", "scripts/compute_norm_stats.py", "--config-name", config, "--num-workers", str(workers)),
        cwd=root,
        env=env,
    )


def build_train(values: Mapping[str, object]) -> CommandSpec:
    root = _openpi_root(str(values["openpi_root"]))
    config = _name(str(values["config_name"]), "训练配置名")
    experiment = _name(str(values["exp_name"]), "实验名")
    checkpoint_root = _output_path(str(values["checkpoint_base_dir"]), "checkpoint 根目录")
    batch_size = _positive(int(values["batch_size"]), "batch size", maximum=65536)
    steps = _positive(int(values["num_train_steps"]), "训练步数")
    fsdp_devices = _positive(int(values["fsdp_devices"]), "FSDP 设备数", maximum=64)
    save_interval = _positive(int(values["save_interval"]), "保存间隔")
    keep_period = _positive(int(values["keep_period"]), "长期保留间隔")
    cuda_devices = str(values["cuda_devices"]).strip()
    if not _CUDA_DEVICES.fullmatch(cuda_devices):
        raise ValueError("CUDA 设备必须写成逗号分隔的非负整数; 例如 2,3")
    if len(cuda_devices.split(",")) != fsdp_devices:
        raise ValueError("CUDA_VISIBLE_DEVICES 的数量必须等于 FSDP 设备数")
    if batch_size % fsdp_devices:
        raise ValueError("batch size 必须能被 FSDP 设备数整除")
    memory_fraction = float(values["memory_fraction"])
    if not 0.1 <= memory_fraction <= 1.0:
        raise ValueError("XLA 显存比例必须在 0.1 到 1.0 之间")
    mode = str(values.get("mode", "overwrite"))
    if mode not in {"overwrite", "resume"}:
        raise ValueError("训练模式只能是 overwrite 或 resume")
    env = _data_environment({key: str(value) for key, value in values.items()})
    env.update(CUDA_VISIBLE_DEVICES=cuda_devices, XLA_PYTHON_CLIENT_MEM_FRACTION=f"{memory_fraction:.4g}")
    return CommandSpec(
        argv=(
            "uv",
            "run",
            "scripts/train.py",
            config,
            f"--exp-name={experiment}",
            f"--checkpoint-base-dir={checkpoint_root}",
            f"--batch-size={batch_size}",
            f"--num-train-steps={steps}",
            f"--fsdp-devices={fsdp_devices}",
            f"--save-interval={save_interval}",
            f"--keep-period={keep_period}",
            f"--{mode}",
        ),
        cwd=root,
        env=env,
        total_units=steps,
    )
