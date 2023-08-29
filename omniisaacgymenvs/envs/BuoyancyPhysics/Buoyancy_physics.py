import torch
from omniisaacgymenvs.envs.BuoyancyPhysics.Utils import *

class BuoyantObject:
    def __init__(self, num_envs, device, water_density, gravity, metacentric_width, metacentric_length, average_buoyancy_force_value, amplify_torque):
            
            self._num_envs = num_envs
            self.device=device
            self.water_density=water_density
            self.gravity= gravity
            self.metacentric_width = metacentric_width
            self.metacentric_length = metacentric_length
            self.archimedes_force_global = torch.zeros((self._num_envs,  3), dtype=torch.float32, device=self.device)
            self.archimedes_torque_global = torch.zeros((self._num_envs,  3), dtype=torch.float32, device=self.device)
            self.archimedes_force_local = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)
            self.archimedes_torque_local = torch.zeros((self._num_envs, 3), dtype=torch.float32, device=self.device)

            #data
            self.average_buoyancy_force_value = average_buoyancy_force_value
            self.amplify_torque = amplify_torque
            return
    
    def compute_archimedes_metacentric_global(self, submerged_volume, rpy):
        
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """

        roll, pitch = rpy[:,0], rpy[:,1]   #roll and pich are given in global frame
        
        #compute buoyancy force
        self.archimedes_force_global[:,2] = - self.water_density * self.gravity * submerged_volume #buoyancy force

        #torques expressed in global frame, size is (num_envs,3)
        """ self.archimedes_torque_global[:,0] = -1 * self.metacentric_width * (torch.sin(roll) * self.archimedes_force_global[:,2])
        self.archimedes_torque_global[:,1] = -1 * self.metacentric_length * (torch.sin(pitch) *  self.archimedes_force_global[:,2]) """

        self.archimedes_torque_global[:,0] = -1 * self.metacentric_width * (torch.sin(roll) * self.average_buoyancy_force_value)  # cannot multiply by the buoyancy force in isaac sim because of the simulation rate (low then high value)
        self.archimedes_torque_global[:,1] = -1 * self.metacentric_length * (torch.sin(pitch) *  self.average_buoyancy_force_value)
        
        #debugging
        #print("self.archimedes_force global: ", self.archimedes_force_global[0,:])
        #print("self.archimedes_torque global: ", self.archimedes_torque_global[0,:])

        return self.archimedes_force_global, self.archimedes_torque_global
    
    
    def compute_archimedes_metacentric_local(self, submerged_volume, rpy, quaternions):
        """This function apply the archimedes force to the center of the boat"""

        """Ideally, this function should not be applied only at the center of the boat, but at the center of the volume submerged underwater.
        In this case, if the boat start to rotate around x and y axis, the part of the boat who isn't underwater anymore have no force except gravity applied,
        it automatically balance the boat. But that would require to create 4 rigid body at each corner of the boat and then check which one of them is underwater.
        """

        #get archimedes global force
        self.compute_archimedes_metacentric_global(submerged_volume, rpy)

        #get rotation matrix from quaternions in world frame, size is (3*num_envs, 3)
        R= getWorldToLocalRotationMatrix(quaternions)

        #print("R:", R[0,:,:])
        
        # Arobot = Rworld * Aworld. Resulting matrix should be size (3*num_envs, 3) * (num_envs,3) =(num_envs,3)
        self.archimedes_force_local = torch.bmm(R.mT,torch.unsqueeze(self.archimedes_force_global, 1).mT) #add batch dimension to tensor and transpose it
        self.archimedes_force_local = self.archimedes_force_local.mT.squeeze(1) #remove batch dimension to tensor

        """ self.archimedes_torque_local = torch.bmm(R.mT,torch.unsqueeze(self.archimedes_torque_global, 1).mT)
        self.archimedes_torque_local = self.archimedes_torque_local.mT.squeeze(1) """
        
        #not sure if torque have to be multiply by the rotation matrix also.
        self.archimedes_torque_local = self.archimedes_torque_global
        
        return torch.hstack([self.archimedes_force_local, self.archimedes_torque_local*self.amplify_torque])
    

    #alternative of archimedes torque
    """
    def stabilize_boat(self,yaws):
        # Roll Stabilizing Force = -k_roll * θ_x, Yaw Stabilizing Force = -k_yaw * θ_z 

        K=5.0 #by hand
        force=torch.zeros((self._num_envs, 3), dtype=torch.float32)
        
        force[:,0] = - K * yaws[:,0]
        force[:,1] = - K * yaws[:,1]

        return force
    """
        


    


