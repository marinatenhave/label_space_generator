import pathlib
import click
import spark_dsg as dsg
from autoabstr.utils.layer_features import Places2DFeatureExtractor


@click.command()
@click.argument("graph_path", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="output location")
@click.option("--max-diff-z", default=1.5, help="max z difference")
def main(graph_path, output, max_diff_z):
    """Add mesh-place connections to graph."""
    graph_path = pathlib.Path(graph_path).expanduser().absolute()
    G = dsg.DynamicSceneGraph.load(graph_path)
    Places2DFeatureExtractor.assign_clusters_to_2d_places(G, max_diff_z)

    if output is None:
        output = graph_path.parent / f"{graph_path.stem}_with_connections.json"
    else:
        output = pathlib.Path(output).expanduser().absolute()

    G.save(output)


if __name__ == "__main__":
    main()
