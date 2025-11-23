"""
Convert 7-D absolute actions to 10-D
"""
import zarr
import numpy as np
from sim_demo_collector.util.transform_util import RotationTransformer

with zarr.open("demos/square_pc/episodes.zarr", 'r') as f:
    actions = f['action'][()]

transformer = RotationTransformer()
trans = actions[:, :3]
rot = actions[:, 3:6]
gripper = actions[:, 6:]
rot = transformer.forward(rot)
actions = np.concatenate([trans, rot, gripper], axis=1)
print("Action shape: ", actions.shape)

with zarr.open("demos/square_pc/episodes.zarr", 'a') as f:
    del f["action"]
    f.create_dataset("action", data=actions, chunks=True)