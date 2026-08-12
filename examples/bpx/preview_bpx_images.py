"""Export a labelled BPX camera preview to verify RGB/BGR interpretation."""

import argparse
import pathlib

import bpx_data
import h5py
import numpy as np
from PIL import Image
from PIL import ImageDraw


def main(
    episode_path: pathlib.Path,
    output: pathlib.Path,
    frame_index: int = 0,
    image_color_order: str = "rgb",
) -> None:
    bpx_data.validate_episode(episode_path)
    with h5py.File(episode_path, "r") as episode:
        frames = []
        for name, key in bpx_data.IMAGE_DATASET_KEYS.items():
            dataset = episode[key]
            if not 0 <= frame_index < dataset.shape[0]:
                raise IndexError(f"frame_index={frame_index} is outside [0, {dataset.shape[0]}).")
            rgb = bpx_data.convert_color(np.asarray(dataset[frame_index]), image_color_order)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 180, 24), fill=(0, 0, 0))
            draw.text((6, 5), name, fill=(255, 255, 255))
            frames.append(image)

    width, height = frames[0].size
    canvas = Image.new("RGB", (width * 2, height * 2))
    for index, image in enumerate(frames):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-path", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("bpx_camera_preview.png"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--image-color-order", choices=("rgb", "bgr"), default="rgb")
    arguments = parser.parse_args()
    main(
        arguments.episode_path,
        arguments.output,
        frame_index=arguments.frame_index,
        image_color_order=arguments.image_color_order,
    )
