import torch
import pytorch3d.transforms

def getWorldToLocalRotationMatrix(quaternions):
    rot_mat = pytorch3d.transforms.quaternion_to_matrix(quaternions)
    return rot_mat

def getLocalLinearVelocities(world_lin_vel, rotWR):
    
    robot_lin_velocity = torch.bmm(rotWR,torch.unsqueeze(world_lin_vel, 1).mT)
    #robot_velocity = torch.matmul(rotWR, world_vel)
    return robot_lin_velocity.mT.squeeze(1) # m/s

def getLocalAngularVelocities(world_ang_vel, rotWR):
    
    robot_ang_velocity = torch.bmm(rotWR,torch.unsqueeze(world_ang_vel, 1).mT)
    #robot_velocity = torch.matmul(rotWR, world_vel)
    return robot_ang_velocity.mT.squeeze(1) # m/s


def CrossProductOperator(A):
    B = torch.zeros([3,3])
    print(A)
    B[0,1] = -A[2]
    B[1,0] = A[2]
    B[0,2] = A[1]
    B[2,0] = -A[1]
    B[2,1] = A[0]
    B[1,2] = -A[0]
    return B
