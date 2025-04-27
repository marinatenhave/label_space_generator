#! /usr/bin/env python3
"""Script to run Hydra against MP3D image datasets."""
import os
import pathlib
import shutil
import signal
import sys
import traceback
import json

import click
import hydra_python as hydra
import numpy as np
import spark_dataset_interfaces as sdi
import torch
from autoabstr.utils.ros_visualizer import ros_visualizer
from label_space_generator import LabelGenerator

WARNING = """
Failed to import semantic_inference!
Install via `roscd semantic_inference && pip install .[openset]`"""

ENABLE_CLIP = True
try:
    import semantic_inference.models as models
except ImportError:
    print(WARNING)
    ENABLE_CLIP = False


WESTPOINT_INFO = {
    "sparkal1": {
        "start_time_s": 220.0,
        "extrinsics": (
            np.array([0.207, -0.052, 0.083]),
            np.array([0.483, -0.514, 0.524, -0.477]),
        ),
    },
    "sparkal2": {
        "start_time_s": 0.0,
        "extrinsics": (
            np.array([-0.0005, 0.1090, 0.1081]),
            np.array([0.5017, -0.5002, 0.4979, -0.5001])
        ),
    },
}

BEACH_INFO = {
    "beach1": {"start_time_s": 71.0, "duration_s": 629.0},
    "beach2": {"start_time_s": 60.0, "duration_s": 340.0},
}


def _check_path(filepath):
    filepath = pathlib.Path(filepath).expanduser().absolute()
    if not filepath.exists():
        raise ValueError(f"Invalid filepath: '{filepath}'")

    return filepath


def _bin_path():
    return pathlib.Path(__file__).absolute().parent

def _repo_path():
    return pathlib.Path(__file__).absolute().parent.parent.parent


def _load_dataset_path(dataset_path, env_name=None):
    if dataset_path is not None:
        return pathlib.Path(dataset_path).expanduser().absolute()

    err_str = "Dataset required! Specify with '-d DATASET_PATH'"
    if env_name is None:
        click.secho(f"{err_str}!", fg="red")
        sys.exit(1)

    var_name = f"{env_name}_PATH"
    err_str += f" or setting environment variable '{var_name}'"
    dataset_path = os.getenv(var_name)
    if dataset_path is None:
        click.secho(f"{err_str}!", fg="red")
        sys.exit(1)

    return pathlib.Path(dataset_path).expanduser().absolute()


def _get_scene_output(output_path, scene_name, force):
    scene_output = output_path / scene_name
    if scene_output.exists():
        if not force:
            click.secho(f"Skipping existing output: '{scene_output}'")
            return None

        click.secho(f"Removing existing output: '{scene_output}'", fg="yellow")
        shutil.rmtree(scene_output)

    return scene_output


def _decompose_pose(pose):
    q_xyzw = pose.rotation.as_quat()
    q_wxyz = [q_xyzw[i] for i in [3, 0, 1, 2]]
    return q_wxyz, pose.translation

import threading
import numpy as np

import threading
import numpy as np

def _run_scene(dataloader, pipeline, max_steps, data_callbacks, label_generator, label_rate=None, scene_output=None):
    label_generator.reset_scene_space()

    last_label_timestamp = None
    min_label_dt_ns = int(1e9 / label_rate) if label_rate else 0

    label_thread_active = False  # 🆕 Track whether label_generator thread is running

    def _step_pipeline(packet):
        nonlocal last_label_timestamp, label_thread_active
        current_ts = packet.timestamp

        print(f"\n🔄 Processing packet at timestamp: {current_ts}")
        print(f"🔍 Packet pose: {packet.pose}")
        print(f"🖼️ Packet color shape: {packet.color.shape if hasattr(packet.color, 'shape') else 'No color shape'}")

        # 🔁 Run Hydra scene graph pipeline every frame
        print(f"🧪 Running hydra step at {current_ts}")
        rotation, translation = _decompose_pose(packet.pose)
        # pipeline.step(
        #     current_ts,
        #     translation,
        #     rotation,
        #     packet.depth,
        #     packet.labels,
        #     packet.color,
        #     **packet.extras,
        # )
        # print("✅ Hydra pipeline step executed.")

        # 🧵 Run LabelGenerator in the background, only if last one finished
        if (last_label_timestamp is None or (current_ts - last_label_timestamp) >= min_label_dt_ns) and not label_thread_active:
            last_label_timestamp = current_ts
            label_thread_active = True  # 🔥 Mark a label thread as active

            def label_thread():
                try:
                    print(f"🏷️ Running label generator at {current_ts}")
                    label_generator.step(np.copy(packet.color))
                finally:
                    nonlocal label_thread_active
                    label_thread_active = False  # 🔥 Mark label thread finished
                    print("✅ Label generator step finished.")

            threading.Thread(target=label_thread, daemon=True).start()

    print("🚀 Starting DataLoader run...")
    sdi.DataLoader.run(
        dataloader,
        _step_pipeline,
        max_steps=max_steps,
        step_mode=False,
        show_progress=True,
        data_callbacks=data_callbacks,
    )

    print("✅ Finished running scene, saving pipeline.")
    pipeline.save()

    if scene_output:
        label_generator.save_scene_label_space(scene_output)

def _resolve_output(output, dataset):
    if output is not None:
        output = pathlib.Path(output).expanduser().absolute()
    else:
        output = _repo_path() / "out" / dataset

    output.mkdir(parents=True, exist_ok=True)
    click.secho(f"Output directory: {output}", fg="green")
    return output


class ClipEncoder:
    """Image feature encoder."""

    def __init__(self, model_name="ViT-L/14"):
        """Load clip."""
        print("Loading clip...")
        config = models.ClipConfig()
        config.model_name = model_name
        self.device = models.default_device()
        self.model = models.ClipWrapper(config).to(self.device)
        self.model.eval()
        self.transform = models.get_image_preprocessor(self.model.input_size).to(
            self.device
        )
        self.center_crop = models.center_crop
        print("Loaded clip!")

    def __call__(self, packet: sdi.InputPacket):
        """Add clip feature encoding to input packet."""

        with torch.no_grad():
            img = torch.from_numpy(packet.color).to(self.device).permute((2, 0, 1))
            img = self.center_crop(self.transform(img), self.model.input_size)
            feature = torch.squeeze(self.model(img.unsqueeze(0)).cpu()).numpy()

        packet.extras["feature"] = feature


@click.group()
def main():
    """Main entry point."""
    pass


@main.command("mp3d")
@click.option("--dataset-path", "-d", default=None, type=click.Path(exists=True))
@click.option("--visualize/--no-visualize", default=True)
@click.option("--zmq-url", default="tcp://127.0.0.1:8001")
@click.option("--max-steps", "-m", default=None, type=int)
@click.option("--use-clip/--no-use-clip", default=True)
@click.option("--force", "-f", is_flag=True, help="overwrite existing scenes.")
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--label-rate", default=None, type=float, help="Throttle label pipeline to LABEL_RATE Hz")

def mp3d(dataset_path, visualize, zmq_url, max_steps, use_clip, force, output, label_rate):
    hydra.set_glog_level(0, 0)
    output = _resolve_output(output, "mp3d")
    dataset_path = _load_dataset_path(dataset_path, "MP3D")
    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]

    label_generator = LabelGenerator()  # 🆕 Instantiate once here

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    data_callbacks = []
    if ENABLE_CLIP and use_clip:
        data_callbacks.append(ClipEncoder())

    with ros_visualizer(visualize, zmq_url=zmq_url) as visualizer:
        click.secho(f"Visualizer running at {visualizer.zmq_url}", fg="green")
        for scene_path in scenes:
            try:
                dataloader = sdi.FileDataLoader(scene_path)
                sensor = hydra.make_camera(**dataloader.intrinsics)
                scene_output = _get_scene_output(output, scene_path.stem, force)
                if scene_output is None:
                    continue

                label_generator.set_autosave_path(scene_output) 

                pipeline = hydra.load_pipeline(
                    sensor,
                    "mp3d",
                    "ade20k_mp3d",
                    config_path=_bin_path() / "config",
                    output_path=scene_output,
                    freeze_global_info=False,
                    zmq_url=zmq_url if visualize else None,
                )

                _run_scene(dataloader, pipeline, max_steps, data_callbacks, label_generator, label_rate)
                label_generator.save_scene_label_space(scene_output)

            except KeyboardInterrupt:
                click.secho(f"⚠️ KeyboardInterrupt detected during scene '{scene_path.stem}'. Saving partial label space...", fg="yellow")
                label_generator.save_scene_label_space(scene_output)
                raise  # re-raise to cleanly quit

            except Exception:
                click.secho(f"Pipeline failed for '{scene_path}'", fg="red")
                click.secho(f"{traceback.format_exc()}", fg="red")

@main.command(name="westpoint")
@click.option("--dataset-path", "-d", default=None, type=click.Path(exists=True))
@click.option("--visualize/--no-visualize", default=True)
@click.option("--zmq-url", default="tcp://127.0.0.1:8001")
@click.option("--max-steps", "-m", default=None, type=int)
@click.option("--use-clip/--no-use-clip", default=True)
@click.option("--force", "-f", is_flag=True, help="overwrite existing scenes.")
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--rate", default=None, type=float, help="Throttle processing to RATE Hz")

def westpoint(dataset_path, visualize, zmq_url, max_steps, use_clip, force, output, rate):
    hydra.set_glog_level(0, 0)
    output = _resolve_output(output, "westpoint")
    dataset_path = _load_dataset_path(dataset_path, "WESTPOINT")
    bags = [
        dataset_path / "sparkal1/kimera_test_2022-06-24-15-48-36.bag",
        dataset_path / "sparkal2/kimera_test_2022-06-24-15-15-51.bag",
    ]

    # NOTE(nathan) something messes with SIGINT and disables it (this puts it back?)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    data_callbacks = []
    if ENABLE_CLIP and use_clip:
        data_callbacks.append(ClipEncoder())

    with ros_visualizer(visualize, zmq_url=zmq_url) as visualizer:
        click.secho(f"Visualizer running at {visualizer.zmq_url}", fg="green")
        for bag_path in bags:
            try:
                robot_name = bag_path.parent.stem
                scene_output = _get_scene_output(output, robot_name, force)
                if scene_output is None:
                    continue

                scene_info = WESTPOINT_INFO[robot_name]
                trajectory = sdi.Trajectory.from_csv(
                    bag_path.with_suffix(".csv"),
                    time_col="#timestamp_kf",
                    pose_cols=["x", "y", "z", "qx", "qy", "qz", "qw"],
                )
                bag = sdi.RosbagDataLoader(
                    bag_path,
                    rgb_topic=f"/{robot_name}/forward/color/image_raw/compressed",
                    rgb_info_topic=f"/{robot_name}/forward/color/camera_info",
                    depth_topic=f"/{robot_name}/forward/depth/image_rect_raw",
                    label_topic="/oneformer/labels/image_raw",
                    trajectory=trajectory,
                    min_separation_s=0.1,
                    start_time_ns=int(scene_info["start_time_s"] * 1.0e9),
                )

                with bag as dataloader:
                    extrinsics = hydra.ExtrinsicsConfig()
                    extrinsics.body_p_sensor = scene_info["extrinsics"][0]
                    q = scene_info["extrinsics"][1]
                    q /= np.linalg.norm(q)
                    extrinsics.body_R_sensor = hydra.Quaterniond(q[0], q[1], q[2], q[3])
                    sensor = hydra.make_camera(
                        **dataloader.intrinsics,
                        max_range=5.0,
                        min_range=0.3,
                        extrinsics=extrinsics,
                    )

                    pipeline = hydra.load_pipeline(
                        sensor,
                        "westpoint",
                        "ade20k_full",
                        config_path=_bin_path() / "config",
                        output_path=scene_output,
                        freeze_global_info=False,
                        zmq_url=zmq_url if visualize else None,
                    )

                    _run_scene(dataloader, pipeline, max_steps, data_callbacks)
            except Exception:
                click.secho(f"Pipeline failed for '{bag_path}'", fg="red")
                click.secho(f"{traceback.format_exc()}", fg="red")


@main.command(name="beach")
@click.option("--dataset-path", "-d", default=None, type=click.Path(exists=True))
@click.option("--visualize/--no-visualize", default=True)
@click.option("--zmq-url", default="tcp://127.0.0.1:8001")
@click.option("--max-steps", "-m", default=None, type=int)
@click.option("--use-clip/--no-use-clip", default=True)
@click.option("--force", "-f", is_flag=True, help="overwrite existing scenes.")
@click.option("--output", "-o", default=None, type=click.Path())
@click.option("--rate", default=None, type=float, help="Throttle processing to RATE Hz")

def beach(dataset_path, visualize, zmq_url, max_steps, use_clip, force, output, rate):
    hydra.set_glog_level(0, 0)
    output = _resolve_output(output, "beach")
    dataset_path = _load_dataset_path(dataset_path, "BEACH")
    bags = [
        dataset_path / "beach1/beach1_2023-03-27-14-04-58.bag",
        dataset_path / "beach2/beach2_2023-03-27-14-57-39.bag",
    ]

    # NOTE(nathan) something messes with SIGINT and disables it (this puts it back?)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    data_callbacks = []
    if ENABLE_CLIP and use_clip:
        data_callbacks.append(ClipEncoder())

    with ros_visualizer(visualize, zmq_url=zmq_url) as visualizer:
        click.secho(f"Visualizer running at {visualizer.zmq_url}", fg="green")
        for bag_path in bags:
            try:
                scene_name = bag_path.parent.stem
                scene_output = _get_scene_output(output, scene_name, force)
                if scene_output is None:
                    continue

                scene_info = BEACH_INFO[scene_name]
                trajectory = sdi.Trajectory.from_csv(
                    bag_path.with_suffix(".csv"),
                    time_col="#timestamp_kf",
                    pose_cols=["x", "y", "z", "qx", "qy", "qz", "qw"],
                )
                bag = sdi.RosbagDataLoader(
                    bag_path,
                    rgb_topic="/cam_d455/color/image_raw/compressed",
                    rgb_info_topic="/cam_d455/color/camera_info",
                    depth_topic="/cam_d455/depth/image_rect_raw",
                    label_topic="/oneformer/labels/image_raw",
                    trajectory=trajectory,
                    min_separation_s=0.1,
                    start_time_ns=int(scene_info["start_time_s"] * 1.0e9),
                )

                with bag as dataloader:
                    extrinsics = hydra.ExtrinsicsConfig()
                    extrinsics.body_p_sensor = np.array(
                        [0.11160737, 0.00136474, -0.00886937]
                    )
                    extrinsics.body_R_sensor = hydra.Quaterniond(
                        0.49944784, 0.50264384, 0.50363997, 0.49421433
                    )
                    sensor = hydra.make_camera(
                        **dataloader.intrinsics,
                        max_range=5.0,
                        min_range=0.3,
                        extrinsics=extrinsics,
                    )

                    pipeline = hydra.load_pipeline(
                        sensor,
                        "beach",
                        "ade20k_full",
                        config_path=_bin_path() / "config",
                        output_path=scene_output,
                        freeze_global_info=False,
                        zmq_url=zmq_url if visualize else None,
                    )

                    _run_scene(dataloader, pipeline, max_steps, data_callbacks)
            except Exception:
                click.secho(f"Pipeline failed for '{bag_path}'", fg="red")
                click.secho(f"{traceback.format_exc()}", fg="red")


if __name__ == "__main__":
    # can run this script with command: 
    # python run_hydra.py mp3d -d '/mnt/c/Users/Marina Ten Have/Downloads/q9vSo1VnCiC-001' --force
    main()