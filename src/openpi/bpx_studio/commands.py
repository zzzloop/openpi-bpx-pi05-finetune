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
    if not value.strip():
        raise ValueError(f"{label}不能为空")
    path = pathlib.Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label}不存在或不是目录: {path}")
    return path


def _output_path(value: str, label: str) -> pathlib.Path:
    if not value.strip():
        raise ValueError(f"{label}不能为空")
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
    for relative in (
        "scripts/compute_norm_stats.py",
        "scripts/serve_policy.py",
        "scripts/train.py",
        "pyproject.toml",
    ):
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
    episodes = values.get("episode_indices")
    episode_args: tuple[str, ...] = ()
    if episodes is not None:
        if not isinstance(episodes, list) or not episodes or any(type(index) is not int or index < 0 for index in episodes):
            raise ValueError("episode 索引必须是非空的非负整数列表")
        episode_args = ("--episodes", *(str(index) for index in sorted(set(episodes))))
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
            *episode_args,
        ),
        cwd=project_root,
    )


def build_normalize(values: Mapping[str, object]) -> CommandSpec:
    root = _openpi_root(str(values["openpi_root"]))
    config = _name(str(values["config_name"]), "训练配置名")
    workers = _positive(int(values.get("workers", 32)), "workers", maximum=256)
    env = _data_environment({key: str(value) for key, value in values.items()})
    # Statistics use NumPy, not model inference. Do not initialize the server's
    # CUDA devices just to create the loader's otherwise-unused JAX sharding.
    # This environment belongs ONLY to the normalization child process.
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    return CommandSpec(
        argv=("uv", "run", "scripts/compute_norm_stats.py", "--config-name", config, "--num-workers", str(workers),
              *_validation_arguments(values, normalize=True)),
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
    checkpoint_limit = values.get("checkpoint_limit")
    retention_arguments = ()
    if checkpoint_limit is not None:
        checkpoint_limit = _positive(int(checkpoint_limit), "Checkpoint 总保留上限", maximum=100)
        if str(values.get("selection_suite", "")).strip() and checkpoint_limit < 2:
            raise ValueError("选模至少需要两个名额：最新可续训模型 + 一个候选")
        retention_arguments = (f"--checkpoint-limit={checkpoint_limit}",)
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
    wandb_mode = str(values.get("wandb_mode", "offline"))
    if wandb_mode not in {"offline", "online", "disabled"}:
        raise ValueError("W&B 模式只能是 offline、online 或 disabled")
    experiment_dir = checkpoint_root / config / experiment
    if mode == "overwrite" and experiment_dir.exists():
        raise ValueError("实验目录已存在，请换一个实验名，或选择断点续训，避免覆盖 checkpoint")
    env = _data_environment({key: str(value) for key, value in values.items()})
    env.update(CUDA_VISIBLE_DEVICES=cuda_devices, XLA_PYTHON_CLIENT_MEM_FRACTION=f"{memory_fraction:.4g}")
    # Optional cloud tracking must not block the default dashboard workflow.
    # CheckpointManager creates this directory before wandb.init is called.
    env.update(WANDB_MODE=wandb_mode, WANDB_DIR=str(experiment_dir))
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
            *retention_arguments,
            f"--{mode}",
            *_validation_arguments(values),
            *_selection_arguments(values),
        ),
        cwd=root,
        env=env,
        total_units=steps,
    )


def _validation_arguments(values, *, normalize=False):
    manifest = str(values.get("validation_manifest", "")).strip()
    if not manifest:
        return ()
    path = pathlib.Path(manifest).expanduser().resolve()
    from openpi.training.validation_split import load_split
    load_split(path)
    if normalize:
        return ("--validation-manifest", str(path))
    interval = _positive(int(values.get("validation_interval", 1000)), "验证间隔")
    batches = _positive(int(values.get("validation_batches", 16)), "验证 batches", maximum=256)
    return (f"--validation-manifest={path}", f"--validation-interval={interval}", f"--validation-batches={batches}")


def _selection_arguments(values):
    arguments = []
    suite = str(values.get("selection_suite", "")).strip()
    if suite:
        if not str(values.get("validation_manifest", "")).strip():
            raise ValueError("多指标选模必须启用分组验证清单")
        path = pathlib.Path(suite).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"固定选模样本清单不存在: {path}")
        keep = _positive(int(values.get("selection_keep", 3)), "候选保留数量", maximum=20)
        denoise = _positive(int(values.get("selection_denoise_steps", 10)), "选模去噪步数", maximum=100)
        arguments.extend((f"--selection-suite={path}", f"--selection-keep={keep}", f"--selection-denoise-steps={denoise}"))
    initial = str(values.get("init_checkpoint", "")).strip()
    if initial:
        if values.get("mode") == "resume":
            raise ValueError("候选初始化是新一轮微调，不能同时使用断点续训 resume")
        path = pathlib.Path(initial).expanduser().resolve()
        if not path.is_file() and not (path / "params").is_dir():
            raise ValueError("初始化路径应为完整 checkpoint 或 recommended.json")
        arguments.append(f"--init-checkpoint={path}")
    return tuple(arguments)


def build_probe_suite(project_root, values):
    config = _name(str(values["config_name"]), "训练配置名")
    output = _output_path(str(values["output_dir"]), "固定评估集输出目录")
    if output.exists():
        raise ValueError("固定评估集不可覆盖，请更换输出目录")
    raw_manifest = str(values.get("validation_manifest", "")).strip()
    if not raw_manifest:
        raise ValueError("先生成并启用分组验证清单，再准备选模评估集")
    manifest = pathlib.Path(raw_manifest).expanduser().resolve()
    from openpi.training.validation_split import load_split
    load_split(manifest)
    arguments = [sys.executable, "scripts/prepare_checkpoint_probes.py", "--config-name", config,
                 "--validation-manifest", str(manifest), "--output-dir", str(output),
                 "--max-cases", str(_positive(int(values.get("max_cases", 16)), "固定场景数", maximum=256))]
    case_spec = str(values.get("case_spec", "")).strip()
    if case_spec:
        path = pathlib.Path(case_spec).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"人工复核指令对文件不存在: {path}")
        arguments.extend(("--case-spec", str(path)))
    env = _data_environment({key: str(value) for key, value in values.items()})
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    return CommandSpec(argv=tuple(arguments), cwd=project_root, env=env)


def build_validation_split(project_root: pathlib.Path, values: Mapping[str, object]) -> CommandSpec:
    dataset = _existing_directory(str(values["dataset_root"]), "LeRobot 数据集")
    output = _output_path(str(values["output_path"]), "验证划分清单")
    if output.exists():
        raise ValueError("验证清单已存在，请复用该清单或更换文件名，不要覆盖")
    fraction = float(values.get("fraction", 0.1))
    if not 0 < fraction < 0.5:
        raise ValueError("验证比例必须在 0 到 0.5 之间")
    argv = [sys.executable, "-m", "openpi.training.validation_split", "--dataset-root", str(dataset),
            "--output", str(output), "--fraction", str(fraction)]
    source = str(values.get("source_manifest", "")).strip()
    if source:
        path = pathlib.Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"源转换 manifest 不存在: {path}")
        argv.extend(("--source-manifest", str(path)))
    elif values.get("independent_episodes", False):
        argv.append("--independent-episodes")
    else:
        raise ValueError("增强数据必须提供源转换 manifest；无增强数据才可确认 episode 相互独立")
    return CommandSpec(argv=tuple(argv), cwd=project_root)


def build_policy_serve(values: Mapping[str, object]) -> CommandSpec:
    root = _openpi_root(str(values["openpi_root"]))
    config = _name(str(values["config_name"]), "推理配置名")
    checkpoint = _existing_directory(str(values["checkpoint_dir"]), "checkpoint 目录")
    for child in ("params", "assets"):
        if not (checkpoint / child).is_dir():
            raise ValueError(f"checkpoint 缺少 {child}/: {checkpoint}")
    cuda_devices = str(values.get("cuda_devices", "2,3")).strip()
    if not _CUDA_DEVICES.fullmatch(cuda_devices):
        raise ValueError("CUDA 设备必须写成逗号分隔的非负整数; 例如 2,3")
    memory_fraction = float(values.get("memory_fraction", 0.96))
    if not 0.1 <= memory_fraction <= 1.0:
        raise ValueError("XLA 显存比例必须在 0.1 到 1.0 之间")
    return CommandSpec(
        argv=(
            "uv",
            "run",
            "scripts/serve_policy.py",
            "policy:checkpoint",
            f"--policy.config={config}",
            f"--policy.dir={checkpoint}",
        ),
        cwd=root,
        env={
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{memory_fraction:.4g}",
        },
    )
