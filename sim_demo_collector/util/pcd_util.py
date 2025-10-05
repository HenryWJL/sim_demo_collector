import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Union


def depth2pcd(
    depth: np.ndarray,
    camera_intrinsic_matrix: np.ndarray,
    bounding_box: Optional[Dict] = None,
    seg_mask: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    bounding_box: {
        'x': (-1, 1),
        'y': (-1, 1),
        'z': (0, 5)
    }
    """
    if len(depth.shape) == 3:
        depth = depth.squeeze(-1)
    height, width = depth.shape
    fx = camera_intrinsic_matrix[0, 0]
    fy = camera_intrinsic_matrix[1, 1]
    cx = camera_intrinsic_matrix[0, 2]
    cy = camera_intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pcd = np.stack([x, y, z], axis=-1)
    # Apply bounding box mask
    bb_mask = np.ones_like(z, dtype=bool)
    if bounding_box is not None:
        bb_mask = (
            (x > bounding_box['x'][0]) &
            (x < bounding_box['x'][1]) &
            (y > bounding_box['y'][0]) &
            (y < bounding_box['y'][1]) &
            (z > bounding_box['z'][0]) &
            (z < bounding_box['z'][1])
        )
    pcd = pcd[bb_mask]
    if seg_mask is not None:
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.squeeze(-1)
        seg_mask = seg_mask[bb_mask]
    return pcd, seg_mask
    

def save_open3d_pcd(
    filename: str,
    pcd_xyz: np.ndarray,
    pcd_rgb: Optional[np.ndarray] = None
) -> None:
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_xyz)
    if pcd_rgb is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_rgb)
    o3d.io.write_point_cloud(filename, pcd_o3d)