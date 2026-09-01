"""Convert aligned BPX HDF5 demonstrations to LeRobot v2.1.

The converter consumes the recorded schema directly:

* ``/observations/qpos``: ``(T, 23)`` actual joint feedback
* ``/action``: ``(T, 23)`` commanded absolute joint targets
* ``/observations/images/{left_eye,right_eye,left_wrist,right_wrist}``

The fixed waist dimensions 0:3 and the right-eye image are intentionally
removed. Each raw jaw pair is summed into one physical gripper value. The
resulting state/action vectors therefore have 18 elements.
"""

import pathlib
import re
import shutil

import bpx_data
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi.policies import bpx_constants

_REPO_ID_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def resolve_output_path(repo_id: str) -> pathlib.Path:
    """Resolve a safe local LeRobot path for an ``organization/dataset`` ID."""
    parts = repo_id.split("/")
    if len(parts) != 2 or any(_REPO_ID_PART.fullmatch(part) is None for part in parts):
        raise ValueError(
            "repo_id must have the safe form 'organization/dataset' using only letters, "
            "digits, dots, underscores, and hyphens."
        )
    cache_root = pathlib.Path(HF_LEROBOT_HOME).resolve()
    output_path = (cache_root / parts[0] / parts[1]).resolve()
    if not output_path.is_relative_to(cache_root):
        raise ValueError(f"Resolved dataset path must remain under {cache_root}.")
    return output_path


def bpx_features() -> dict:
    image_feature = {
        "dtype": "image",
        "shape": bpx_constants.BPX_SOURCE_IMAGE_SHAPE,
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.left_eye": dict(image_feature),
        "observation.images.left_wrist": dict(image_feature),
        "observation.images.right_wrist": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (bpx_constants.BPX_EFFECTIVE_DIM,),
            "names": list(bpx_constants.BPX_EFFECTIVE_JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (bpx_constants.BPX_EFFECTIVE_DIM,),
            "names": list(bpx_constants.BPX_EFFECTIVE_JOINT_NAMES),
        },
    }


def main(
    raw_dir: pathlib.Path,
    repo_id: str = "bpx/bpx_blocks",
    *,
    episodes: list[int] | None = None,
    fps: int = bpx_constants.BPX_FPS,
    default_task: str = bpx_constants.BPX_DEFAULT_PROMPT,
    image_color_order: str = "rgb",
    overwrite: bool = False,
    push_to_hub: bool = False,
    private: bool = True,
) -> None:
    """Convert every ``episode_*.hdf5`` in ``raw_dir``."""
    if fps != bpx_constants.BPX_FPS:
        raise ValueError(
            f"BPX demonstrations were confirmed at {bpx_constants.BPX_FPS} Hz; "
            f"refusing fps={fps} to prevent temporal mislabelling."
        )
    if image_color_order not in {"rgb", "bgr"}:
        raise ValueError("image_color_order must be 'rgb' or 'bgr'.")

    files = bpx_data.select_episodes(bpx_data.discover_episodes(raw_dir), episodes)
    summaries = [bpx_data.validate_episode(path) for path in files]
    explicit_failures = [summary.path for summary in summaries if summary.success is False]
    if explicit_failures:
        names = ", ".join(path.name for path in explicit_failures)
        raise ValueError(f"Refusing explicitly unsuccessful episode(s): {names}.")

    output_path = resolve_output_path(repo_id)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists at {output_path}. Choose another repo_id or pass --overwrite."
            )
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="bpx20260310",
        fps=fps,
        features=bpx_features(),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
    )

    for path, summary in zip(files, summaries, strict=True):
        with h5py.File(path, "r") as episode:
            qpos = np.asarray(episode[bpx_data.QPOS_KEY][:], dtype=np.float32)
            actions = np.asarray(episode[bpx_data.ACTION_KEY][:], dtype=np.float32)
            states_18 = np.asarray(bpx_constants.to_effective_joints(qpos), dtype=np.float32)
            actions_18 = np.asarray(bpx_constants.to_effective_joints(actions), dtype=np.float32)
            task = bpx_data.read_task(episode, default_task)
            left_eye = episode[bpx_data.IMAGE_DATASET_KEYS["left_eye"]]
            left_wrist = episode[bpx_data.IMAGE_DATASET_KEYS["left_wrist"]]
            right_wrist = episode[bpx_data.IMAGE_DATASET_KEYS["right_wrist"]]

            for frame_index in range(summary.frames):
                dataset.add_frame(
                    {
                        "observation.images.left_eye": bpx_data.convert_color(left_eye[frame_index], image_color_order),
                        "observation.images.left_wrist": bpx_data.convert_color(
                            left_wrist[frame_index], image_color_order
                        ),
                        "observation.images.right_wrist": bpx_data.convert_color(
                            right_wrist[frame_index], image_color_order
                        ),
                        "observation.state": states_18[frame_index],
                        "action": actions_18[frame_index],
                        "task": task,
                    }
                )
        dataset.save_episode()
        print(f"Saved {path.name}: {summary.frames} frames, task={task!r}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["bpx", "bimanual", "real-robot"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Converted {len(files)} episode(s) to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
