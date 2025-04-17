"""Run against kitti."""
import click
import subprocess
import pathlib
import rospy
import std_srvs.srv
import shutil
import time


def _log_banner(message, char="*", width=80):
    inner = f"{char} {message.center(width - 4)} {char}"
    wrapper = char * len(inner)
    return "\n".join([wrapper, inner, wrapper])


@click.command()
@click.argument("kitti_path", type=click.Path(exists=True))
@click.argument("output_path")
@click.option("-s", "--sequences", multiple=True, default=[0, 2, 3, 4, 5, 6, 7, 9, 10])
@click.option("-v", "--visualize", is_flag=True, help="run visualizer")
@click.option("-r", "--rate", default=0.4, type=float, help="rate to run at")
@click.option("-d", "--duration", default=None, type=float, help="sim duration")
@click.option("--overwrite", is_flag=True, help="re-run sequences")
def main(kitti_path, output_path, sequences, visualize, rate, duration, overwrite):
    """Run Hydra against all Kitti-360 sequences."""
    kitti_path = pathlib.Path(kitti_path).expanduser().absolute()
    output_path = pathlib.Path(output_path).expanduser().absolute()
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for sequence in sequences:
        output_name = f"kitti360_s{sequence:02d}"
        click.echo(_log_banner(f"Running sequence {output_name}"))

        sequence_output = output_path / output_name
        if sequence_output.exists() and not overwrite:
            click.secho(f"\nskippping completed sequence {output_name}\n", fg="green")
            continue

        hydra_args = [
            shutil.which("roslaunch"),
            "hydra_ros",
            "kitti_360.launch",
            f"dsg_output_dir:={output_path}",
            f"dsg_output_prefix:={output_name}",
            "start_visualizer:=true" if visualize else "start_visualizer:=false",
        ]
        hydra_pipe = subprocess.Popen(
            hydra_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        # TODO(nathan) we could wait for shutdown service technically...

        kitti_args = [
            shutil.which("roslaunch"),
            "kitti360_publisher",
            "Kitti360.launch",
            f"directory:={kitti_path}",
            f"rate:={rate}",
            f"sequence:={sequence:02d}",
        ]

        if duration:
            kitti_args += [f"end:={duration}"]

        subprocess.run(kitti_args)
        shutdown_serv = rospy.ServiceProxy(
            "/hydra_ros_node/shutdown", std_srvs.srv.Empty
        )
        shutdown_serv()

        hydra_pipe.wait()


if __name__ == "__main__":
    main()
