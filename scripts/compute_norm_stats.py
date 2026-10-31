"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

# Startup diagnostics must run before the heavy third-party imports below.
# ruff: noqa: E402

import dataclasses
import faulthandler
import os


def _stage(message: str) -> None:
    print(f"[BPX NORM] {message}", flush=True)
    if __name__ == "__main__":
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(60, repeat=True)


if __name__ == "__main__":
    faulthandler.enable()
    _stage("初始化 Python/ML 依赖；若同阶段超过60秒，将自动打印执行堆栈")

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.validation_split as _validation_split
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    _stage(f"加载 LeRobot 数据集及校验时间戳：{data_config.repo_id}")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    _stage(f"数据加载完成：已选择 {len(dataset)} 个参考帧；准备动作变换")
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    sampler = _data_loader.create_task_balanced_sampler(data_config, len(dataset), seed=0)
    _stage(
        f"初始化数据加载器/设备：workers={num_workers}；"
        f"JAX_PLATFORMS={os.getenv('JAX_PLATFORMS', 'auto')}"
    )
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    num_workers: int | None = None,
    validation_manifest: str | None = None,
):
    """Compute normalization statistics for a training config.

    Args:
        config_name: Registered training configuration name.
        max_frames: Optional upper bound on the number of sampled frames.
        num_workers: Optional torch DataLoader worker override. Leaving this unset
            preserves the worker count from the training configuration. Increasing
            it only parallelizes input loading; it does not change the selected
            frames, batch size, transforms, or statistics algorithm.
    """
    _stage(f"解析 Config 及训练/验证划分：{config_name}")
    config = _config.get_config(config_name)
    config = _validation_split.configure(dataclasses.replace(config, validation_manifest=validation_manifest))
    effective_num_workers = config.num_workers if num_workers is None else num_workers
    if effective_num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = _validation_split.select_data_config(config, data_config)
    selected = data_config.episode_indices
    _stage(
        f"数据配置就绪：{len(selected) if selected is not None else '全部'} 个训练 episode；"
        f"workers={effective_num_workers}"
    )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            effective_num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    _stage(f"开始归一化统计：{num_batches} batches；等待 workers 返回首批数据")
    first_batch = True
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        if first_batch:
            if __name__ == "__main__":
                faulthandler.cancel_dump_traceback_later()
            first_batch = False
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    _stage("全部 batches 统计完成；汇总均值、标准差与分位数")
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    _stage(f"写入归一化统计：{output_path}")
    print(f"Writing stats to: {output_path}", flush=True)
    normalize.save(output_path, norm_stats)
    _stage(f"归一化文件保存完成：{output_path}/norm_stats.json；等待进程退出确认")


if __name__ == "__main__":
    try:
        tyro.cli(main)
    finally:
        faulthandler.cancel_dump_traceback_later()
