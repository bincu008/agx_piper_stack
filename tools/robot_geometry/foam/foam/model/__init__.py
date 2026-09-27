from dataclasses import dataclass
from json import JSONEncoder, JSONDecoder

import numpy as np
from numpy.typing import NDArray


@dataclass(slots = True)
class Sphere:
    origin: NDArray
    radius: float

    def __init__(self, x: float, y: float, z: float, r: float, offset: NDArray | None = None):
        self.origin = np.array([x, y, z])
        if offset is not None:
            self.origin += offset

        self.radius = r

    def offset(self, offset: NDArray):
        self.origin += offset


@dataclass(slots = True)
class Spherization:
    spheres: list[Sphere]
    mean_error: float
    best_error: float
    worst_error: float

    def __len__(self) -> int:
        return len(self.spheres)

    def __lt__(self, other) -> bool:
        return self.mean_error < other.mean_error and self.best_error < other.best_error and self.worst_error < other.worst_error

    def offset(self, offset: NDArray):
        for sphere in self.spheres:
            sphere.offset(offset)



class SphereEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Sphere):
            return {'origin': list(obj.origin), 'radius': obj.radius}
        if isinstance(obj, Spherization):
            return {
                'mean': obj.mean_error,
                'best': obj.best_error,
                'worst': obj.worst_error,
                'spheres': obj.spheres
                }

        return JSONEncoder.default(self, obj)


class SphereDecoder(JSONDecoder):

    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(self, object_hook = self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if 'origin' in dct and 'radius' in dct:
            return Sphere(*dct['origin'], dct['radius']) # type: ignore

        if 'mean' in dct and 'best' in dct and 'worst' in dct and 'spheres' in dct:
            return Spherization(dct['spheres'], dct['mean'], dct['best'], dct['worst'])

        return dct
