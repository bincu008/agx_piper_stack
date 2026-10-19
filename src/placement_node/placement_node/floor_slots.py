"""Reusable floor lattice and footprint-to-cell matching, independent of ROS."""

from dataclasses import dataclass
import hashlib

import cv2
import numpy as np


def layout_key(destination):
    """Invalidate cached slots when the measured destination boundary changes."""
    points = np.asarray(destination, np.float32)
    hull = cv2.convexHull(np.ascontiguousarray(points[:, :2])).reshape(-1, 2)
    quantized = np.unique(np.round(hull / .005).astype(np.int32), axis=0)
    return hashlib.sha256(quantized.tobytes() +
                          np.round(np.median(points[:, 2]) / .005).tobytes()).hexdigest()


@dataclass
class FloorSlots:
    """Frozen adaptive rectangles or legacy squares aligned with the destination."""

    origin: np.ndarray
    axes: np.ndarray
    cells: np.ndarray
    size: float = .125
    gap: float = .005
    bounds: np.ndarray | None = None
    cell_gaps: np.ndarray | None = None
    large_cell: tuple[int, int] | None = None
    working_bounds: np.ndarray | None = None

    @classmethod
    def build(cls, boundary, clearance=.002, size=.125, gap=.005):
        """Keep only complete squares inside the usable destination boundary."""
        boundary = np.asarray(boundary, np.float32)
        corners = cv2.boxPoints(cv2.minAreaRect(boundary))
        edges = np.roll(corners, -1, axis=0) - corners
        axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        axis = axis / np.linalg.norm(axis)
        if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
            axis = -axis
        axes = np.array([axis, [-axis[1], axis[0]]], np.float32)
        local = boundary @ axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        count = np.floor((high-low-2*clearance+gap)/(size+gap)).astype(int)
        if np.any(count <= 0):
            return cls(low @ axes, axes, np.empty((0, 2), np.int32), size, gap)
        span = count * (size+gap)-gap
        origin = ((low+high-span)*.5) @ axes
        polygon = cv2.convexHull(boundary).reshape(-1, 2)
        cells = []
        for y in range(count[1]):
            for x in range(count[0]):
                vertices = (np.array([[0, 0], [size, 0], [size, size], [0, size]]) +
                            np.array([x, y])*(size+gap)) @ axes + origin
                if all(cv2.pointPolygonTest(polygon, tuple(map(float, p)), True) >=
                       clearance-1e-6 for p in vertices):
                    cells.append((x, y))
        return cls(origin, axes, np.asarray(cells, np.int32).reshape(-1, 2), size, gap)

    def to_dict(self):
        """Serialize immutable geometry through shared search diagnostics."""
        return {'origin': self.origin.tolist(), 'axes': self.axes.tolist(),
                'cells': self.cells.tolist(), 'size': self.size, 'gap': self.gap,
                **({'large_cell': list(self.large_cell)} if self.large_cell is not None else {}),
                **({'working_bounds': self.working_bounds.tolist()}
                   if self.working_bounds is not None else {}),
                **({'bounds': self.bounds.tolist(),
                    'cell_gaps': self.cell_gaps.tolist() if self.cell_gaps is not None else None}
                   if self.bounds is not None else {})}

    @classmethod
    def from_dict(cls, data):
        """Restore a frozen grid without recomputing its alignment."""
        return cls(np.asarray(data['origin']), np.asarray(data['axes']),
                   np.asarray(data['cells'], np.int32).reshape(-1, 2),
                   data['size'], data['gap'],
                   np.asarray(data['bounds']) if 'bounds' in data else None,
                   np.asarray(data['cell_gaps']) if data.get('cell_gaps') is not None else None,
                   tuple(data['large_cell']) if data.get('large_cell') is not None else None,
                   np.asarray(data['working_bounds']) if 'working_bounds' in data else None)

    def vertices(self, cell):
        """Return cell corners in the destination XY frame."""
        if self.bounds is not None:
            index = np.flatnonzero(np.all(self.cells == cell, axis=1))[0]
            x0, y0, x1, y1 = self.bounds[index]
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ self.axes + self.origin
        return (np.array([[0, 0], [self.size, 0], [self.size, self.size], [0, self.size]]) +
                np.asarray(cell)*(self.size+self.gap)) @ self.axes + self.origin

    def territory_vertices(self, cell):
        """Include half the local gap so groups can span physical tray support."""
        vertices = self.vertices(cell)
        gap = self.gap
        if self.cell_gaps is not None:
            gap = self.cell_gaps[np.flatnonzero(np.all(self.cells == cell, axis=1))[0]]
        center = vertices.mean(axis=0)
        local = (vertices-center) @ self.axes.T
        return (local+np.sign(local)*gap/2) @ self.axes+center

    @staticmethod
    def unique_pairs(index):
        """Encode integer pairs for fast one-dimensional stencil deduplication."""
        if not len(index):
            return set()
        low = index.min(axis=0)
        width = int(index[:, 1].max()-low[1]+1)
        codes = np.unique((index[:, 0]-low[0])*width+index[:, 1]-low[1])
        return {(int(code//width+low[0]), int(code % width+low[1])) for code in codes}

    @staticmethod
    def reservation_mask(centered_source, rotation, translation):
        """Reserve projected geometry at 5 mm resolution with one-cell dilation."""
        xy = (centered_source @ rotation.T + translation)[:, :2]
        mask = FloorSlots.unique_pairs(np.floor(xy/.005).astype(np.int32))
        return {(x+dx, y+dy) for x, y in mask
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def contains_in_cells(self, centered_source, rotation, translation, cells):
        """Check achieved geometry against its slot group, not the ideal pose."""
        xyz = np.asarray(centered_source) @ rotation.T + translation
        if not len(xyz) or not np.isfinite(xyz).all():
            return False
        local = (xyz[:, :2]-self.origin) @ self.axes.T
        used = set(map(tuple, cells))
        if len(used) == 1:
            cell = next(iter(used))
            if self.bounds is not None:
                index = np.flatnonzero(np.all(self.cells == cell, axis=1))
                if not len(index):
                    return False
                low, high = self.bounds[index[0], :2], self.bounds[index[0], 2:]
            else:
                low = np.asarray(cell)*(self.size+self.gap)
                high = low+self.size
            return bool(np.all(local.min(axis=0) >= low-1e-8) and
                        np.all(local.max(axis=0) <= high+1e-8))
        # Sample the completed projected interior too, so a boundary cannot
        # surround an unavailable cell without reserving it.
        hull = cv2.convexHull(local.astype(np.float32)).reshape(-1, 2)
        low, high = hull.min(axis=0), hull.max(axis=0)
        x, y = np.meshgrid(np.arange(low[0], high[0]+.001, .002),
                           np.arange(low[1], high[1]+.001, .002))
        samples = np.column_stack((x.ravel(), y.ravel()))
        inside = np.array([cv2.pointPolygonTest(hull, tuple(map(float, p)), False) >= 0
                           for p in samples])
        touched = self._touched_local(np.vstack((local, samples[inside])), 0)
        return bool(touched and touched <= used and (-1, -1) not in touched)

    def touched(self, footprint, clearance=0):
        """Rasterize the footprint into pitch cells, reserving intervening gaps."""
        local = (np.asarray(footprint)-self.origin) @ self.axes.T
        return self._touched_local(local, clearance)

    def _touched_local(self, local, clearance, extents=None):
        """Reuse grid coordinates while preserving point-level gap membership."""
        if not len(local):
            return set()
        if self.bounds is not None:
            # A footprint wholly inside a physical slot belongs to that slot.
            # Wide gap territories beside irregular leftovers must not steal
            # space from an adjacent square or inflate its reservation.
            low, high = extents if extents is not None else (
                local.min(axis=0)-clearance, local.max(axis=0)+clearance)
            for cell, bounds in zip(self.cells, self.bounds):
                if (np.all(low >= bounds[:2]-1e-8) and
                        np.all(high <= bounds[2:]+1e-8)):
                    return {tuple(map(int, cell))}
            points = np.vstack([local + d for d in ((0, 0), (-clearance, -clearance),
                               (-clearance, clearance), (clearance, -clearance),
                               (clearance, clearance))]) if clearance else local
            found = set()
            covered = np.zeros(len(points), bool)
            for index, (cell, bounds) in enumerate(zip(self.cells, self.bounds)):
                gap = self.cell_gaps[index] if self.cell_gaps is not None else self.gap
                lower, upper = bounds[:2]-gap/2-1e-8, bounds[2:]+gap/2+1e-8
                # Scalar columns avoid four large Nx2 reduction temporaries.
                inside = ((points[:, 0] >= lower[0]) & (points[:, 1] >= lower[1]) &
                          (points[:, 0] <= upper[0]) & (points[:, 1] <= upper[1]))
                if inside.any():
                    found.add(tuple(map(int, cell)))
                    covered |= inside
            if not covered.all():
                found.add((-1, -1))
            return found
        index = np.floor((local+self.gap*.5)/(self.size+self.gap)).astype(int)
        # Sampling includes the completed footprint interior from the heightmap.
        if clearance > 0:
            index = np.vstack([np.floor((local + delta + self.gap*.5) /
                                        (self.size+self.gap)).astype(int)
                               for delta in ((-clearance, -clearance),
                                             (-clearance, clearance),
                                             (clearance, -clearance),
                                             (clearance, clearance))] + [index])
        return self.unique_pairs(index)

    def trials(self, footprint, reserved=(), clearance=.002, phase_offsets=(0.,),
               single_only=False):
        """Slide a footprint stencil over available groups, smallest group first."""
        if self.bounds is not None:
            return self._rectangle_trials(footprint, reserved, clearance, phase_offsets,
                                          single_only)
        available = set(map(tuple, self.cells)) - set(map(tuple, reserved))
        if not available:
            return []
        local = np.asarray(footprint) @ self.axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        span = np.maximum(1, np.ceil(
            (high-low+2*clearance+self.gap) / (self.size+self.gap)-1e-6).astype(int))
        center_offset = (low+high)*.5
        results = []
        # Cache one stencil per phase; lattice translations only shift its IDs.
        phases = []
        for dx in phase_offsets:
            for dy in phase_offsets:
                phase = np.array([dx, dy])
                position = ((span*(self.size+self.gap)-self.gap)*.5-center_offset+phase)
                translation = position @ self.axes+self.origin
                stencil = self.touched(footprint+translation, clearance)
                if span.max() == 1:
                    placed = local+position
                    if np.any(placed.min(axis=0) < clearance-1e-6) or np.any(
                            placed.max(axis=0) > self.size-clearance+1e-6):
                        continue
                phases.append((translation, stencil))
        for x, y in self.cells:
            for translation, stencil in phases:
                touched = {(a+int(x), b+int(y)) for a, b in stencil}
                if touched and touched <= available:
                    results.append((translation+np.array([x, y])*(self.size+self.gap) @
                                    self.axes, touched))
        results.sort(key=lambda item: (
            len(item[1]), min(x+y for x, y in item[1]), float(np.linalg.norm(item[0]))))
        return results

    def fit_priority(self, cells):
        """Prefer fewer occupied slots, breaking count ties by reserved area."""
        used = set(map(tuple, cells))
        if self.bounds is None:
            area = len(used)*self.size*self.size
        else:
            indices = np.array([tuple(cell) in used for cell in self.cells])
            dimensions = self.bounds[indices, 2:]-self.bounds[indices, :2]
            area = float(np.sum(np.prod(dimensions, axis=1)))
        return len(used), round(area, 8)

    def _rectangle_trials(self, footprint, reserved, clearance, phase_offsets,
                          single_only=False):
        """Match rectangular cell groups without moving established boundaries."""
        available = set(map(tuple, self.cells))-set(map(tuple, reserved))
        local = np.asarray(footprint) @ self.axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        extent, middle = high-low+2*clearance, (high+low)/2
        ends_x, ends_y = np.unique(self.bounds[:, 2]), np.unique(self.bounds[:, 3])
        gaps = self.gap if self.cell_gaps is None else self.cell_gaps
        if np.ndim(gaps) == 1:
            gaps = gaps[:, None]
        territory_low = self.bounds[:, :2]-gaps/2
        territory_high = self.bounds[:, 2:]+gaps/2
        territory_min, territory_max = territory_low.min(axis=0), territory_high.max(axis=0)
        indices = {tuple(cell): i for i, cell in enumerate(self.cells)}
        result, seen = [], set()
        for index, (cell, bounds) in enumerate(zip(self.cells, self.bounds)):
            if tuple(cell) not in available:
                continue
            ex = ends_x[ends_x >= bounds[0]+extent[0]-1e-8]
            ey = ends_y[ends_y >= bounds[1]+extent[1]-1e-8]
            if not len(ex) or not len(ey):
                continue
            group_high = np.array([ex[0], ey[0]])
            position = (bounds[:2]+group_high)/2-middle
            for dx in phase_offsets:
                for dy in phase_offsets:
                    translated = position + [dx, dy]
                    key = tuple(np.round(translated, 7))
                    if key in seen:
                        continue
                    seen.add(key)
                    world = translated @ self.axes+self.origin
                    footprint_low = low+translated-clearance
                    footprint_high = high+translated+clearance
                    if single_only and not np.any(
                            np.all(footprint_low >= self.bounds[:, :2]-1e-8, axis=1) &
                            np.all(footprint_high <= self.bounds[:, 2:]+1e-8, axis=1)):
                        continue
                    # Any extremum outside all cell territories necessarily
                    # yields the outside sentinel; reject before scanning points.
                    if (np.any(footprint_low < territory_min-1e-8) or
                            np.any(footprint_high > territory_max+1e-8)):
                        continue
                    overlaps = (np.all(footprint_high >= territory_low-1e-8, axis=1) &
                                np.all(footprint_low <= territory_high+1e-8, axis=1))
                    overlaps[index] = False
                    # A strictly contained bounding rectangle proves every point
                    # belongs to this cell. Preserve the point-level path whenever
                    # another cell's gap territory could also be touched.
                    if (np.all(footprint_low > bounds[:2]+1e-7) and
                            np.all(footprint_high < bounds[2:]-1e-7) and
                            not overlaps.any()):
                        used = {tuple(cell)}
                    else:
                        used = self._touched_local(local+translated, clearance,
                                                   (footprint_low, footprint_high))
                    if not used or not used <= available:
                        continue
                    if len(used) == 1:
                        box = self.bounds[indices[next(iter(used))]]
                        if np.any(low+translated < box[:2]+clearance-1e-8) or np.any(
                                high+translated > box[2:]-clearance+1e-8):
                            continue
                    result.append((world, used))
        result.sort(key=lambda item: (self.fit_priority(item[1]),
                                      float(np.linalg.norm(item[0]))))
        return result

    @staticmethod
    def _largest_rectangle(mask):
        """Return the largest axis-aligned rectangle of usable raster cells."""
        heights = np.zeros(mask.shape[1], int)
        best, area = None, 0
        for y, row in enumerate(mask):
            heights = np.where(row, heights+1, 0)
            stack = []
            for x, height in enumerate(np.append(heights, 0)):
                start = x
                while stack and stack[-1][1] > height:
                    left, h = stack.pop()
                    if h*(x-left) > area:
                        area, best = h*(x-left), (left, y-h+1, x, y+1)
                    start = left
                if not stack or stack[-1][1] < height:
                    stack.append((start, int(height)))
        return best

    @staticmethod
    def _square_region(length, depth, maximum, minimum):
        """Keep two square rows; add rows only at maximum square size."""
        if min(length, depth) < minimum-1e-8:
            return [], 0, 0, 0
        full_rows = int(np.floor(depth/maximum+1e-8))
        rows = (full_rows if full_rows >= 3 and length >= maximum-1e-8 else
                2 if depth >= 2*minimum-1e-8 else 1)
        size = min(maximum, depth/rows, length)
        row_gap = (depth-rows*size)/(rows-1) if rows > 1 else 0
        columns = max(1, int(np.floor(length/size+1e-8)))
        remainder = max(0, length-columns*size)
        widths = [size]*columns
        if remainder >= minimum-1e-8:
            # Only a leftover strip becomes rectangular.
            widths.append(remainder)
            column_gap = 0
        elif columns > 1:
            column_gap = remainder/(columns-1)
        else:
            column_gap = 0
        return widths, rows, size, (column_gap, row_gap)

    @classmethod
    def build_adaptive(cls, boundary, clearance=.002, maximum=.125,
                       minimum=.06, resolution=.002,
                       surface_mask=None, surface_origin=None, surface_resolution=None,
                       large_square_scale=1.5):
        """
        Cover the largest inscribed rectangle and non-overlapping extensions.

        Rows follow the long axis. Geometry is frozen for later occupancy updates.
        The inscribed-rectangle calculation is conservative at raster resolution.
        """
        if (not np.isfinite([minimum, maximum, clearance, resolution,
                             large_square_scale]).all() or
                resolution <= 0 or clearance < 0 or
                not (0 < minimum <= maximum)):
            raise ValueError('INVALID_ADAPTIVE_SLOT_DIMENSIONS')
        if large_square_scale != 0 and large_square_scale <= 1:
            raise ValueError('LARGE_SQUARE_SCALE_MUST_EXCEED_ONE_OR_BE_ZERO')
        polygon = np.ascontiguousarray(boundary, np.float32)
        corners = cv2.boxPoints(cv2.minAreaRect(polygon))
        edges = np.roll(corners, -1, axis=0)-corners
        axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        axis /= np.linalg.norm(axis)
        if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
            axis = -axis
        axes = np.array([axis, [-axis[1], axis[0]]], np.float32)
        local = polygon @ axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        shape = np.ceil((high-low)/resolution).astype(int)+1
        if np.any(shape > 600) or np.any(shape < 2):
            raise ValueError('ADAPTIVE_SLOT_SURFACE_SIZE_INVALID')
        mask = np.zeros((shape[1], shape[0]), np.uint8)
        cv2.fillPoly(mask, [np.rint((local-low)/resolution).astype(np.int32)], 1)
        if surface_mask is not None:
            yy, xx = np.indices(mask.shape)
            world = (np.stack((xx, yy), axis=-1)*resolution+low) @ axes
            index = np.rint((world-np.asarray(surface_origin))/surface_resolution).astype(int)
            inside = ((index[:, :, 0] >= 0) & (index[:, :, 1] >= 0) &
                      (index[:, :, 0] < surface_mask.shape[1]) &
                      (index[:, :, 1] < surface_mask.shape[0]))
            sampled = np.zeros_like(mask)
            sampled[inside] = surface_mask[index[:, :, 1][inside], index[:, :, 0][inside]]
            mask &= sampled
        distance = cv2.distanceTransform(
            np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]*resolution
        mask = (distance >= clearance+resolution).astype(np.uint8)
        regions, attempts = [], 0
        while mask.any() and len(regions) < 64 and attempts < 128:
            attempts += 1
            box = cls._largest_rectangle(mask)
            if box is None:
                break
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = 0
            extent = np.array([x1-x0-1, y1-y0-1])*resolution
            if np.min(extent) < minimum:
                continue
            regions.append((np.array([x0, y0])*resolution,
                            np.array([x1-1, y1-1])*resolution))
        working_bounds = np.asarray([np.r_[lo, hi] for lo, hi in regions]).reshape(-1, 4)
        cells, boxes, gaps, row_id, large_cell = [], [], [], 0, None
        while regions:
            lo, hi = regions.pop(0)
            widths, rows, size, region_gaps = cls._square_region(
                hi[0]-lo[0], hi[1]-lo[1], maximum, minimum)
            if not widths:
                continue
            gap_x, gap_y = region_gaps
            large_size = size*large_square_scale
            make_large = (large_cell is None and large_square_scale > 1 and rows >= 2 and
                          hi[0]-lo[0]-size >= large_size-1e-8 and
                          hi[1]-lo[1] >= large_size-1e-8)
            if make_large:
                # Keep the first column's ordinary squares, including two rows.
                # Reserve one square at the spare end before dividing leftovers.
                large_low = hi-large_size
                regions[:0] = [
                    (np.array([lo[0]+size, lo[1]]), np.array([hi[0], large_low[1]])),
                    (np.array([lo[0]+size, large_low[1]]),
                     np.array([large_low[0], hi[1]]))]
                widths, gap_x = [size], 0
            y = lo[1]
            for _ in range(rows):
                x = lo[0]
                for col, width in enumerate(widths):
                    cells.append((col, row_id))
                    boxes.append((x, y, x+width, y+size))
                    gaps.append((gap_x, gap_y))
                    x += width+gap_x
                row_id += 1
                y += size+gap_y
            if make_large:
                large_cell = (0, row_id)
                cells.append(large_cell)
                boxes.append((*large_low, *hi))
                gaps.append((0, 0))
                row_id += 1
        return cls(low @ axes, axes, np.asarray(cells, np.int32).reshape(-1, 2),
                   maximum, float(np.max(gaps)) if gaps else 0,
                   np.asarray(boxes, np.float64).reshape(-1, 4), np.asarray(gaps), large_cell,
                   working_bounds)
