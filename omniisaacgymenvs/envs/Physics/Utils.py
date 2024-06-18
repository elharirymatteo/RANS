import torch

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
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

def getWorldToLocalRotationMatrix(quaternions):
    rot_mat = quaternion_to_matrix(quaternions)
    return rot_mat


def getLocalLinearVelocities(world_lin_vel, rotWR):

    robot_lin_velocity = torch.bmm(rotWR, torch.unsqueeze(world_lin_vel, 1).mT)
    # robot_velocity = torch.matmul(rotWR, world_vel)
    return robot_lin_velocity.mT.squeeze(1)  # m/s


def getLocalAngularVelocities(world_ang_vel, rotWR):

    robot_ang_velocity = torch.bmm(rotWR, torch.unsqueeze(world_ang_vel, 1).mT)
    # robot_velocity = torch.matmul(rotWR, world_vel)
    return robot_ang_velocity.mT.squeeze(1)  # m/s


def CrossProductOperator(A):
    B = torch.zeros([3, 3])
    print(A)
    B[0, 1] = -A[2]
    B[1, 0] = A[2]
    B[0, 2] = A[1]
    B[2, 0] = -A[1]
    B[2, 1] = A[0]
    B[1, 2] = -A[0]
    return B
