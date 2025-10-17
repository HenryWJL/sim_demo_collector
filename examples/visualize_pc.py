import zarr
import time
import open3d as o3d


def visualize_pc():
    with zarr.open("demos/lift_pc/episodes.zarr", 'r') as f:
        pc_seq = f['frontview_pc'][()]
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence", width=960, height=540)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_seq[0])
    vis.add_geometry(pc)

    for _, raw_pc in enumerate(pc_seq):
        pc.points = o3d.utility.Vector3dVector(raw_pc)
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    vis.destroy_window()


if __name__ == "__main__":
    visualize_pc()