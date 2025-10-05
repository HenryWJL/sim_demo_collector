import fpsample
import numpy as np
import open3d as o3d
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map


def depth2pcd(
    depth,
    intrinsic_matrix,
    bounding_box = None,
    seg_mask = None
):
    """
    bounding_box: {
        'x': (-1, 1),
        'y': (-1, 1),
        'z': (0, 5)
    }
    """
    height, width = depth.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pcd = np.stack([x, y, z], axis=-1)

    mask = np.ones_like(z, dtype=bool)
    if bounding_box is not None:
        mask = (
            (x > bounding_box['x'][0]) &
            (x < bounding_box['x'][1]) &
            (y > bounding_box['y'][0]) &
            (y < bounding_box['y'][1]) &
            (z > bounding_box['z'][0]) &
            (z < bounding_box['z'][1])
        )
    pcd = pcd[mask]
    if seg_mask is not None:
        seg_mask = seg_mask[mask]
    return pcd, seg_mask


# init env
camera_names = ['frontview', 'robot0_eye_in_hand']
H = 84
W = 84
# Lift
bounding_boxes = {
    'frontview': {
        'x': [-0.5, 0.5],
        'y': [-0.4, 2],
        'z': [1, 2.35]
    },
    'robot0_eye_in_hand': {
        'x': [-0.5, 0.5],
        'y': [-0.5, 0.5],
        'z': [0, 0.5]
    },
}
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config['control_delta'] = False
controller_config['uncouple_pos_ori'] = False
env = suite.make(
    "NutAssemblySquare",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    ignore_done=True,
    use_camera_obs=True,
    use_object_obs=False,
    camera_names=camera_names,
    camera_heights=H,
    camera_widths=W,
    camera_depths=True,
    control_freq=20,
    horizon=200,
    camera_segmentations="element",
)

obs = env.reset()
pcds = {camera_name: [] for camera_name in camera_names}
seg_masks = {camera_name: [] for camera_name in camera_names}
for i in range(5):
    action = np.random.rand(*env.action_spec[0].shape)
    obs, reward, done, info = env.step(action)

    # get point cloud
    for camera_name in camera_names:
        depth = obs[f'{camera_name}_depth']
        depth = get_real_depth_map(env.sim, depth).squeeze(-1)

        seg_img = obs[f"{camera_name}_segmentation_element"].squeeze(-1)
        arm_geom_names = env.robots[0].robot_model.visual_geoms
        gripper_geom_names = env.robots[0].gripper.visual_geoms
        robot_geom_names = arm_geom_names + gripper_geom_names
        robot_geom_ids = [env.sim.model.geom_name2id(n) for n in robot_geom_names]
        seg_mask = np.isin(seg_img, robot_geom_ids).astype(np.uint8)

        intrinsic_matrix = get_camera_intrinsic_matrix(
            env.sim, camera_name, H, W
        )
        pcd, seg_mask = depth2pcd(depth, intrinsic_matrix, bounding_boxes[camera_name], seg_mask)
        idx = fpsample.bucket_fps_kdline_sampling(pcd, 512, h=3)
        pcd = pcd[idx]
        seg_mask = seg_mask[idx]
        pcds[camera_name].append(pcd)
        seg_masks[camera_name].append(seg_mask)

"""IMPORTANT
DO NOT visualize point clouds using open3d during rollouts, as this may cause
the simulated environment to fail in rendering depth images.
"""
for camera_name in camera_names:
    for pcd, seg_mask in zip(pcds[camera_name], seg_masks[camera_name]):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        colors = np.zeros_like(pcd)
        colors[seg_mask == 1] = [1, 0, 0]   # red = robot
        colors[seg_mask == 0] = [0, 1, 0]   # green = environment
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(f"{camera_name}_pcd.ply", pcd_o3d)