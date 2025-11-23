import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Union

# Adapted from https://github.com/charlesq34/pointnet/blob/master/part_seg/test.py#L82
def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc


def depth2pc(
    depth: np.ndarray,
    camera_intrinsic_matrix: np.ndarray,
    bounding_box: Optional[Dict] = None,
    seg_mask: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    pc = np.stack([x, y, z], axis=-1)
    # Apply bounding box mask
    bb_mask = np.ones_like(z, dtype=bool)
    if bounding_box is not None:
        bb_mask = (
            (np.all(pc[..., :3] > bounding_box['lower_bound'], axis=-1))
            & (np.all(pc[..., :3] < bounding_box['upper_bound'], axis=-1))
        )
    pc = pc[bb_mask]
    if seg_mask is not None:
        if len(seg_mask.shape) == 3:
            seg_mask = seg_mask.squeeze(-1)
        seg_mask = seg_mask[bb_mask]
    return pc, seg_mask
    

def save_open3d_pc(
    filename: str,
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None
) -> None:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(filename, pc)