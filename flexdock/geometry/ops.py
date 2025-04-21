import math
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rigid_transform_kabsch_numpy(A, B, check_rmsds: bool = False):
    # independent from below
    # returns rotation to apply to B to get to A
    # A, B are Nx3

    assert A.shape == B.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=0, keepdims=True)
    centroid_B = np.mean(B, axis=0, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # compute rotation matrix
    rot, rssd = Rotation.align_vectors(Am, Bm)
    R = rot.as_matrix()
    rmsd = rssd / np.sqrt(A.shape[0])
    t = centroid_A - (R @ centroid_B.T).T

    # check
    B2 = (R @ B.T).T + t
    rmsd2 = np.sqrt(np.mean(np.sum((A - B2) ** 2, axis=1)))
    if check_rmsds:
        assert np.allclose(rmsd, rmsd2, rtol=3.0e-4, atol=1.0e-4), f"{rmsd} vs {rmsd2}"

    return R, t, rmsd


def rigid_transform_kabsch(A, B, as_numpy: bool = False):
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)

    # A and B are * x N x 3
    assert A.shape[-2] == B.shape[-2]
    assert A.shape[-1] == 3

    centroid_A = A.mean(axis=-2, keepdims=True)
    centroid_B = B.mean(axis=-2, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.swapaxes(-1, -2) @ Bm
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.swapaxes(-1, -2) @ U.swapaxes(-1, -2)
    invert_mask = torch.linalg.det(R) < 0
    SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=R.device))
    R[invert_mask] = (Vt.swapaxes(-1, -2)[invert_mask] @ SS.unsqueeze(0)) @ U.swapaxes(
        -1, -2
    )[invert_mask]

    t = (-R @ centroid_A.swapaxes(-1, -2) + centroid_B.swapaxes(-1, -2)).squeeze(-1)

    if as_numpy:
        return R.cpu().numpy(), t.cpu().numpy()

    return R, t


def rigid_transform_kabsch_batch(A, B, batch):
    # A and B are N x 3
    assert A.shape == B.shape
    assert len(A.shape) == 2
    assert len(batch.shape) == 1
    assert A.shape[0] == batch.shape[0]
    assert A.shape[1] == 3

    ligand_sizes = torch.bincount(batch)
    centroid_A = (
        torch.zeros(max(batch) + 1, 3, device=A.device).index_add(0, batch, A)
        / ligand_sizes[:, None]
    )
    centroid_B = (
        torch.zeros(max(batch) + 1, 3, device=B.device).index_add(0, batch, B)
        / ligand_sizes[:, None]
    )

    Am, _ = to_dense_batch(A - centroid_A[batch], batch)
    Bm, _ = to_dense_batch(B - centroid_B[batch], batch)

    H = Am.swapaxes(1, 2) @ Bm
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.swapaxes(1, 2) @ U.swapaxes(1, 2)

    invert_mask = torch.linalg.det(R) < 0
    SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=R.device))
    R[invert_mask] = (Vt.swapaxes(1, 2)[invert_mask] @ SS.unsqueeze(0)) @ U.swapaxes(
        1, 2
    )[invert_mask]

    t = (-R @ centroid_A.unsqueeze(-1) + centroid_B.unsqueeze(-1)).squeeze(-1)

    return R, t


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert (
        math.fabs(torch.linalg.det(R) - 1) < 3e-3
    )  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t
