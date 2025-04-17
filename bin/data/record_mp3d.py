#!/usr/bin/env python3
"""Record a dataset of images from habitat (specifically mp3d)."""

import multiprocessing as mp
import pathlib
import shutil

import click
import imageio.v3
import numpy as np
import spark_dataset_interfaces as sdi
import tqdm
import yaml


class DatasetLogger:
    """Save dataset to disk."""

    def __init__(self, output_path):
        """Construct a rosbag."""
        self._output_path = pathlib.Path(output_path).expanduser().absolute()
        self._output_path.mkdir(parents=True, exist_ok=False)
        (self._output_path / "color").mkdir(parents=True, exist_ok=False)
        (self._output_path / "depth").mkdir(parents=True, exist_ok=False)
        (self._output_path / "labels").mkdir(parents=True, exist_ok=False)

        self._pose_file = None
        self._index = 0

    def __enter__(self):
        """Start pose file."""
        self._pose_file = (self._output_path / "poses.csv").open("w")
        self._pose_file.write("timestamp_ns,tx,ty,tz,qw,qx,qy,qz\n")
        self._index = 0
        return self

    def __exit__(self, *args):
        """Stop pose file."""
        self._pose_file.close()

    def _color_suffix(self, index):
        return f"rgb_{index:07d}.png"

    def _depth_suffix(self, index):
        return f"depth_{index:07d}.tiff"

    def _labels_suffix(self, index):
        return f"labels_{index:07d}.png"

    def _image_path(self, camera, index=None):
        index = index if index is not None else self._index
        if camera == "color":
            suffix = self._color_suffix(index)
        elif camera == "depth":
            suffix = self._depth_suffix(index)
        elif camera == "labels":
            suffix = self._labels_suffix(index)
        else:
            raise RuntimeError(f"Invalid camera type {camera}")

        return self._output_path / camera / suffix

    def save(self, timestamp, pose, depth, labels, rgb):
        """Save output."""
        assert self._pose_file is not None

        t = pose.translation
        # get rotation in wxyz order
        q_xyzw = pose.rotation.as_quat()
        q = [q_xyzw[i] for i in [3, 0, 1, 2]]
        self._pose_file.write(
            f"{timestamp},{t[0]},{t[1]},{t[2]},{q[0]},{q[1]},{q[2]},{q[3]}\n"
        )

        imageio.v3.imwrite(self._image_path("color"), rgb)
        imageio.v3.imwrite(self._image_path("depth"), depth)
        imageio.v3.imwrite(self._image_path("labels"), labels.astype(np.uint8))
        self._index += 1


@click.group()
def main():
    """Commands to record data from habitat."""
    pass


@main.command(name="scene")
@click.argument("scene_path", type=click.Path(exists=True))
@click.argument("trajectory_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("--progress/--no-progress", default=True, help="show progress bar")
@click.option("--max-images", "-m", default=None, type=int, help="max images to record")
def record(scene_path, trajectory_path, output_path, force, progress, max_images):
    """Create a dataset of inputs from MP3D for Hydra."""
    from hydra_python.simulators import habitat

    output_path = pathlib.Path(output_path).expanduser().absolute()
    if output_path.exists():
        click.secho(f"[WARNING]: output {output_path} already exists", fg="yellow")
        if not force:
            click.confirm(f"remove {output_path}?", abort=True, default=False)

        shutil.rmtree(output_path)

    data = habitat.HabitatInterface(scene_path)
    poses = sdi.Trajectory.from_csv(trajectory_path)

    with DatasetLogger(output_path) as recorder:
        with (output_path / "camera_info.yaml").open("w") as fout:
            fout.write(yaml.dump(data.sensor))

        total = len(poses) if max_images is None else min(len(poses), max_images)
        for index, stamped_pose in enumerate(
            tqdm.tqdm(poses, disable=not progress, total=total)
        ):
            if max_images is not None and index >= max_images:
                break

            timestamp, pose = stamped_pose
            data.set_pose(timestamp, pose.matrix())
            recorder.save(timestamp, pose, data.depth, data.labels, data.rgb)


def _scene_trampoline(ctx, **kwargs):
    ctx.invoke(record, **kwargs)


@main.command("scenes")
@click.argument("mp3d_path", type=click.Path(exists=True))
@click.argument("trajectory_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--progress/--no-progress", default=True, help="show progress bar")
@click.option("--max-images", "-m", default=None, type=int, help="max images to record")
@click.pass_context
def record_scenes(ctx, mp3d_path, trajectory_path, output, progress, max_images):
    """Make a static dataset of mp3d."""
    mp3d_path = pathlib.Path(mp3d_path).expanduser().absolute()
    trajectory_path = pathlib.Path(trajectory_path).expanduser().absolute()
    output = pathlib.Path(output).expanduser().absolute()

    mp3d_scenes = [x for x in mp3d_path.glob("**/*.glb")]
    for scene in mp3d_scenes:
        kwargs = {
            "scene_path": str(scene),
            "trajectory_path": str(trajectory_path / f"{scene.stem}.csv"),
            "output_path": str(output / scene.stem),
            "max_images": max_images,
            "progress": progress,
            "force": True,
        }
        proc = mp.Process(target=_scene_trampoline, args=(ctx,), kwargs=kwargs)
        proc.start()
        proc.join()


if __name__ == "__main__":
    main()
