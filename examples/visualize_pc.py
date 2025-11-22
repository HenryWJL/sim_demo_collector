import zarr
import time
import numpy as np
import open3d as o3d


def visualize_pc():
    with zarr.open("demos/robosuite_square_pc/episodes.zarr", 'r') as f:
        pc_seq = f['frontview_pc'][()]
        pc_masks = f['frontview_pc_mask'][()]
        actions = f['action'][()]
        episode_ends = f['meta/episode_ends'][()]
        print("Point cloud shape: ", pc_seq.shape)
        print("Action shape: ", actions.shape)
        print("Episode ends: ", episode_ends)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence", width=960, height=540)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_seq[0])
    colors = np.zeros_like(pc_seq[0])
    colors[pc_masks[0] == 1] = [1, 0, 0]   # red = robot
    colors[pc_masks[0] == 0] = [0, 1, 0]   # green = environment
    pc.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pc)

    for raw_pc, pc_mask in zip(pc_seq, pc_masks):
        pc.points = o3d.utility.Vector3dVector(raw_pc)
        colors = np.zeros_like(raw_pc)
        colors[pc_mask == 1] = [1, 0, 0]   # red = robot
        colors[pc_mask == 0] = [0, 1, 0]   # green = environment
        pc.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    vis.destroy_window()


if __name__ == "__main__":
    visualize_pc()