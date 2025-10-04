import torch
import numpy as np
from typing import Optional, Union, List
from scipy.spatial.transform import Rotation as R


def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    return np.concatenate([col0, col1], axis=-1)


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    a1 = rot_6d[..., 0: 3]
    a2 = rot_6d[..., 3: 6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    mat = np.stack((b1, b2, b3), axis=-1)
    return mat

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/rotation_transformer.py
class RotationTransformer:
    valid_reps = [
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "matrix",
    ]

    def __init__(
        self,
        from_rep: Optional[str] = "axis_angle",
        to_rep: Optional[str] = "rotation_6d", 
        from_convention: Optional[str] = None,
        to_convention: Optional[str] = None
    ) -> None:
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            if from_rep == "axis_angle":
                forward_funcs.append(lambda x: R.from_rotvec(x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_rotvec())
            elif from_rep == "euler_angles":
                forward_funcs.append(lambda x: R.from_euler(from_convention, x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_euler(from_convention))
            elif from_rep == "quaternion":
                forward_funcs.append(lambda x: R.from_quat(x).as_matrix())
                inverse_funcs.append(lambda x: R.from_matrix(x).as_quat())
            elif from_rep == "rotation_6d":
                forward_funcs.append(rotation_6d_to_matrix)
                inverse_funcs.append(matrix_to_rotation_6d)

        if to_rep != "matrix":
            if to_rep == "axis_angle":
                forward_funcs.append(lambda x: R.from_matrix(x).as_rotvec())
                inverse_funcs.append(lambda x: R.from_rotvec(x).as_matrix())
            elif to_rep == "euler_angles":
                forward_funcs.append(lambda x: R.from_matrix(x).as_euler(to_convention))
                inverse_funcs.append(lambda x: R.from_euler(to_convention, x).as_matrix())
            elif to_rep == "quaternion":
                forward_funcs.append(lambda x: R.from_matrix(x).as_quat())
                inverse_funcs.append(lambda x: R.from_quat(x).as_matrix())
            elif to_rep == "rotation_6d":
                forward_funcs.append(matrix_to_rotation_6d)
                inverse_funcs.append(rotation_6d_to_matrix)

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: List) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, torch.Tensor):
            x_ = x.detach().cpu().numpy()
        for func in funcs:
            x_ = func(x_)
        if isinstance(x, torch.Tensor):
            return torch.from_numpy(x_)
        return x_

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)