import torch
import math

class VirtualPlatform:
    def __init__(self, num_envs, max_thrusters, device):
        self._num_envs = num_envs
        self._device = device
        self._max_thrusters = max_thrusters

        self.transforms2D = torch.zeros((num_envs, max_thrusters, 3, 3), device=self._device, dtype=torch.float32)
        self.current_transforms = torch.zeros((num_envs, max_thrusters, 5), device=self._device, dtype=torch.float32)
        self.action_masks = torch.zeros((num_envs, max_thrusters), device=self._device, dtype=torch.long)        
        self.create_unit_vector()

    def create_unit_vector(self):
        tmp_x = torch.ones((self._num_envs, self._max_thrusters, 1), device=self._device, dtype=torch.float32)
        tmp_y = torch.zeros((self._num_envs, self._max_thrusters, 1), device=self._device, dtype=torch.float32)
        self.unit_vector = torch.cat([tmp_x, tmp_y], dim=-1)

    def project_forces(self, forces):
        # Split transforms into translation and rotation
        R = self.transforms2D[:,:,:2,:2].reshape(-1,2,2)
        T = self.transforms2D[:,:,2,:2].reshape(-1,2)
        # Create a zero tensor to add 3rd dimmension
        zero = torch.zeros((T.shape[0], 1), device=self._device, dtype=torch.float32)
        # Generate positions
        positions = torch.cat([T,zero], dim=-1)
        # Project forces
        force_vector = self.unit_vector * forces.view(-1,self._max_thrusters,1)
        rotated_forces = torch.matmul(R.reshape(-1,2,2), force_vector.view(-1,2,1))
        projected_forces = torch.cat([rotated_forces[:,:,0], zero],dim=-1)

        return positions, projected_forces
    
    def randomize_thruster_state(self, env_ids, num_resets):
        self.generate_base_platforms(num_resets, env_ids)

    def generate_base_platforms(self, num_envs, env_ids):
        random_offset = torch.ones((num_envs), device=self._device).view(-1,1).expand(num_envs, 8) * math.pi / 4
        thrust_offset = torch.arange(4, device=self._device).repeat_interleave(2).expand(num_envs, 8)/4 * math.pi * 2
        thrust_90 = (torch.arange(2, device=self._device).repeat(4).expand(num_envs,8) * 2 - 1) * math.pi / 2 
        mask = torch.ones((num_envs, 8), device=self._device)

        theta = random_offset + thrust_offset + thrust_90
        theta2 = random_offset + thrust_offset

        self.transforms2D[env_ids,:,0,0] = torch.cos(theta) * mask
        self.transforms2D[env_ids,:,0,1] = torch.sin(-theta) * mask
        self.transforms2D[env_ids,:,1,0] = torch.sin(theta) * mask
        self.transforms2D[env_ids,:,1,1] = torch.cos(theta) * mask
        self.transforms2D[env_ids,:,2,0] = torch.cos(theta2) * 0.5 * mask
        self.transforms2D[env_ids,:,2,1] = torch.sin(theta2) * 0.5 * mask
        self.transforms2D[env_ids,:,2,2] = 1 * mask

        self.action_masks[env_ids, :] = 1 - mask.long()

        self.current_transforms[env_ids, :, 0] = torch.cos(theta) * mask
        self.current_transforms[env_ids, :, 1] = torch.sin(-theta) * mask
        self.current_transforms[env_ids, :, 2] = torch.cos(theta2) * 0.5 * mask
        self.current_transforms[env_ids, :, 3] = torch.sin(theta2) * 0.5 * mask
        self.current_transforms[env_ids, :, 4] = 1 * mask

    def visualize(self, env_ids, show=False, save=False, save_path=None):
        pass
        #from matplotlib import pyplot as plt
        #R = self.transforms2D[env_ids, :, :2,:2]
        #O = self.transforms2D[env_ids, :,  2,:2]

        #torch.ones()
        #x = torch.ones([num_envs, 8, 1])
        #y = torch.zeros([num_envs, 8, 1])
        #xy = torch.cat([x,y],dim=-1)

        #uv = torch.matmul(R.reshape(-1,2,2),xy.view(num_envs*8,2,1))
        #plt.quiver(Op[0,:,0], Op[0,:,1], uvp[:,0,0], uvp[:,1,0], scale=2, scale_units='xy', angles='xy')
        #plt.xlim([-1,1])
        #plt.ylim([-1,1])
        #plt.show()

