from pathlib import Path

from fire import Fire

from foam import *


def main(
        filename: str = "spherized.urdf",
    ):
    print("[")
    for x, y, z, r in get_urdf_spheres(load_urdf(Path(filename))):
        print(f"({x:6.5f},{y:6.5f},{z:6.5f},{r:6.5f}),")
    print("]")


if __name__ == "__main__":
    Fire(main)
