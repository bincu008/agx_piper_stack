import random
from json import load as jsload

import matplotlib as mpl
from fire import Fire
from trimesh.creation import icosphere
from trimesh.exchange.load import load_mesh
from trimesh.transformations import translation_matrix
from trimesh.viewer import SceneViewer

from foam import *


def main(mesh: str, spheres: str | None = None, depth: int = 1):
    mesh_filepath = Path(mesh)
    if not mesh_filepath.exists:
        raise RuntimeError(f"Path {mesh} does not exist!")

    scene = Scene([load_mesh_file(mesh_filepath)])

    if spheres:
        sphere_filepath = Path(spheres)
        if not sphere_filepath.exists:
            raise RuntimeError(f"Path {spheres} does not exist!")

        with open(sphere_filepath, 'r') as json_file:
            spherization = jsload(json_file, cls = SphereDecoder)

        if depth > len(spherization):
            raise RuntimeError(f"Depth {depth} greater than available ({len(data)})!")

        cm = mpl.colormaps['viridis']
        for sphere in spherization[depth].spheres:
            sphere_mesh = icosphere(radius = sphere.radius)
            sphere_mesh.visual.face_colors = [255 * c for c in cm(random.uniform(0, 1))][:3] + [100]
            scene.add_geometry(sphere_mesh, transform = translation_matrix(sphere.origin))

    SceneViewer(scene)


if __name__ == "__main__":
    Fire(main)
