import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

TYPE_FORMATS = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
}


def remove_nan_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be an Nx3 array")
    mask = np.isfinite(points).all(axis=1)
    return points[mask].astype(np.float32)


def transform_points(points: np.ndarray, transform: Optional[Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if transform is None:
        return points

    if isinstance(transform, dict):
        rotation = transform.get("rotation")
        translation = transform.get("translation")
        if rotation is not None:
            points = transform_points(points, rotation)
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float32)
            if translation.shape != (3,):
                raise ValueError("translation must be a 3-element vector")
            points = points + translation
        return points

    transform_arr = np.asarray(transform, dtype=np.float32)
    if transform_arr.shape == (4, 4):
        pad = np.ones((points.shape[0], 1), dtype=np.float32)
        homogeneous = np.concatenate([points, pad], axis=1)
        transformed = homogeneous.dot(transform_arr.T)
        return transformed[:, :3].astype(np.float32)

    if transform_arr.shape == (3, 3):
        return points.dot(transform_arr.T).astype(np.float32)

    if transform_arr.shape == (3,):
        return (points + transform_arr).astype(np.float32)

    raise ValueError(
        f"Unsupported transform shape {transform_arr.shape}. "
        "Expected (3,), (3,3), or (4,4)."
    )


def extract_roi(
    points: np.ndarray,
    roi_x: Tuple[float, float],
    roi_z: Tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be an Nx3 array")

    x_min, x_max = roi_x
    z_min, z_max = roi_z

    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask].astype(np.float32)


def save_pointcloud(points: np.ndarray, path: str) -> None:
    points = remove_nan_points(points)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]

    with target.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for x, y, z in points:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def load_pointcloud(path: str) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Point cloud file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.load(source).astype(np.float32)
    if suffix == ".ply":
        return read_ply(source)

    return np.loadtxt(source, dtype=np.float32)


def read_ply(path: str) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"PLY file not found: {source}")

    with source.open("rb") as fh:
        header_lines: List[str] = []
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("Unexpected end of PLY header")
            line = raw.decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        format_line = next(
            (line for line in header_lines if line.startswith("format ")), None
        )
        if format_line is None:
            raise ValueError("PLY header missing format declaration")

        vertex_count = 0
        properties: List[Tuple[str, str]] = []
        current_element: Optional[str] = None

        for line in header_lines:
            if line.startswith("element "):
                _, element_name, element_count_str = line.split()
                current_element = element_name
                if element_name == "vertex":
                    vertex_count = int(element_count_str)
                else:
                    current_element = None
                continue

            if current_element == "vertex" and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("PLY list properties are not supported")
                properties.append((parts[1], parts[2]))

        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if "ascii" in format_line:
            return _read_ply_ascii(fh, properties, vertex_count)
        if "binary_little_endian" in format_line:
            return _read_ply_binary(fh, properties, vertex_count, "<")
        if "binary_big_endian" in format_line:
            return _read_ply_binary(fh, properties, vertex_count, ">")

        raise ValueError(f"Unsupported PLY format: {format_line}")


def _read_ply_ascii(
    fh: Any, properties: List[Tuple[str, str]], vertex_count: int
) -> np.ndarray:
    names = [name for _, name in properties]
    indices = [_index_of(names, axis) for axis in ("x", "y", "z")]

    points: List[Tuple[float, float, float]] = []
    for _ in range(vertex_count):
        line = fh.readline().decode("utf-8").strip()
        if not line:
            raise ValueError("Unexpected end of ASCII PLY data")
        parts = line.split()
        points.append(
            (
                float(parts[indices[0]]),
                float(parts[indices[1]]),
                float(parts[indices[2]]),
            )
        )

    return np.asarray(points, dtype=np.float32)


def _read_ply_binary(
    fh: Any,
    properties: List[Tuple[str, str]],
    vertex_count: int,
    endian: str,
) -> np.ndarray:
    fmt = endian + "".join(_format_char(prop_type) for prop_type, _ in properties)
    record_size = struct.calcsize(fmt)
    names = [name for _, name in properties]
    indices = [_index_of(names, axis) for axis in ("x", "y", "z")]

    points: List[Tuple[float, float, float]] = []
    for _ in range(vertex_count):
        raw = fh.read(record_size)
        if len(raw) != record_size:
            raise ValueError("Unexpected end of binary PLY data")
        values = struct.unpack(fmt, raw)
        points.append(
            (
                float(values[indices[0]]),
                float(values[indices[1]]),
                float(values[indices[2]]),
            )
        )

    return np.asarray(points, dtype=np.float32)


def _format_char(prop_type: str) -> str:
    if prop_type not in TYPE_FORMATS:
        raise ValueError(f"Unsupported PLY property type: {prop_type}")
    return TYPE_FORMATS[prop_type]


def _index_of(names: List[str], key: str) -> int:
    if key not in names:
        raise ValueError(f"PLY file missing required property: {key}")
    return names.index(key)


# def load_pointcloud_o3d(path: str,downsample_ratio: float = 1.0) -> "o3d.geometry.PointCloud":
#     """
#     Load a point cloud file via Open3D (PLY / PCD and other formats).
#     Optionally apply random downsampling.
#     Normals are estimated if not present.
#     Requires open3d to be installed.
#     """
#     if not _O3D_AVAILABLE:
#         raise ImportError("open3d is required for load_pointcloud_o3d().")

#     source = Path(path)
#     if not source.exists():
#         raise FileNotFoundError(f"Point cloud file not found: {source}")

#     pcd = o3d.io.read_point_cloud(str(source))

#     if downsample_ratio < 1.0:
#         pcd = pcd.random_down_sample(sampling_ratio=downsample_ratio)

#     if not pcd.has_normals():
#         pcd.estimate_normals()

#     return pcd


# def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
#     """
#     Voxel-grid downsample a numpy point cloud via Open3D.
#     Returns a numpy (N, 3) float32 array.
#     """
#     if not _O3D_AVAILABLE:
#         raise ImportError("open3d is required for voxel_downsample().")

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
#     pcd = pcd.voxel_down_sample(voxel_size)
#     return np.asarray(pcd.points, dtype=np.float32)


# def remove_statistical_outliers(
#     points: np.ndarray,
#     nb_neighbors: int = 20,
#     std_ratio: float = 2.0,
# ) -> np.ndarray:
#     """
#     Remove statistical outliers from a numpy point cloud via Open3D.
#     Returns a numpy (N, 3) float32 array.
#     """
#     if not _O3D_AVAILABLE:
#         raise ImportError("open3d is required for remove_statistical_outliers().")

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
#     pcd, _ = pcd.remove_statistical_outlier(
#         nb_neighbors=nb_neighbors, std_ratio=std_ratio
#     )
#     return np.asarray(pcd.points, dtype=np.float32)


# def estimate_normals(
#     pcd: "o3d.geometry.PointCloud",
#     radius: float = 0.1,
#     max_nn: int = 30,
# ) -> "o3d.geometry.PointCloud":
#     """
#     Estimate normals for an Open3D PointCloud in-place and return it.
#     """
#     if not _O3D_AVAILABLE:
#         raise ImportError("open3d is required for estimate_normals().")

#     pcd.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
#     )
#     return pcd