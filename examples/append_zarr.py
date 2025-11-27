import zarr
import numpy as np

# Open the two roots
dst_path = "demos/robosuite_square.zarr"   # destination
src_path = "demos/robosuite_square_pc_new.zarr"   # source

def append_array(dst: zarr.Array, src):
    n_old = dst.shape[0]
    n_add = src.shape[0]
    dst.resize(n_old + n_add, *dst.shape[1:])
    dst[n_old:n_old + n_add] = src[:]


def find_first_array(group: zarr.Group):
    for k, v in group.items():
        if isinstance(v, zarr.Array):
            return v
        elif isinstance(v, zarr.Group):
            arr = find_first_array(v)
            if arr is not None:
                return arr
    return None


def append_zarr_with_episode_ends(src_path, dst_path):
    dst = zarr.open(dst_path, "a")
    src = zarr.open(src_path, "r")

    # ----------------------------------------------------------
    # 1. Find first array to determine offset
    # ----------------------------------------------------------
    sample_array = find_first_array(dst)
    if sample_array is None:
        raise RuntimeError("Destination Zarr contains no arrays to determine offset.")
    offset = sample_array.shape[0]

    # ----------------------------------------------------------
    # 2. Recursively append arrays (shift episode_ends)
    # ----------------------------------------------------------
    def rec(dst_group, src_group):
        for name, src_item in src_group.items():
            if isinstance(src_item, zarr.Array):
                if name == "episode_ends" and dst_group.path.endswith("meta"):
                    shifted = src_item[:] + offset
                    append_array(dst_group[name], shifted)
                else:
                    append_array(dst_group[name], src_item)
            else:
                rec(dst_group[name], src_item)

    rec(dst, src)


append_zarr_with_episode_ends(src_path, dst_path)