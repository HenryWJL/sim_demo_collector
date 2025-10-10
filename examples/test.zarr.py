import zarr
import numpy as np
from numcodecs import Blosc


class DataRecorder:
    def __init__(self, zarr_path, shape, dtype=np.float32):
        self.root = zarr.open(zarr_path, mode='a')  # 'a' = read/write, create if needed
        self.compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
        
        if 'data' not in self.root:
            # Create a resizable array along axis 0 (for appending)
            self.data_array = self.root.create(
                'data',
                shape=(0, *shape),  # empty initially
                chunks=(1, *shape),  # tune chunk size based on your use case
                dtype=dtype,
                compressor=self.compressor,
                overwrite=False,
                fill_value=0,
                maxshape=(None, *shape)  # allow appending along first axis
            )
        else:
            self.data_array = self.root['data']
    
    def save(self, data: np.ndarray):
        """Append new data to Zarr efficiently."""
        if data.ndim == len(self.data_array.shape) - 1:
            data = np.expand_dims(data, axis=0)  # ensure (1, *shape)
        
        n_old = self.data_array.shape[0]
        n_new = n_old + data.shape[0]
        self.data_array.resize(n_new, axis=0)
        self.data_array[n_old:n_new] = data
