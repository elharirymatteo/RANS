import torch

class Dynamics:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device= device
        self.thrusters=torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.cmd_updated = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.Reset()

    def update(self, cmd, dt):
      raise NotImplementedError()

    def Reset(self):
        self.cmd_updated[:,:] = 0.0

class DynamicsZeroOrder(Dynamics):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        return
    def update(self, cmd):
        return cmd

class DynamicsFirstOrder(Dynamics):
    def __init__(self, num_envs, device, timeConstant, dt, numberOfPointsForInterpolation, interpolationPointsFromRealData, coeff_neg_commands, coeff_pos_commands, cmd_lower_range, cmd_upper_range ):

        super().__init__(num_envs, device)
        self.tau = timeConstant
        self.idx_matrix = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.dt=dt

        #interpolate
        self.commands = torch.linspace(cmd_lower_range, cmd_upper_range, steps=len(interpolationPointsFromRealData), device=self.device)
        self.numberOfPointsForInterpolation = numberOfPointsForInterpolation
        self.interpolationPointsFromRealData = torch.tensor(interpolationPointsFromRealData, device=self.device)

        #forces
        self.thruster_forces_before_dynamics = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        
        #lsm
        self.coeff_neg_commands=torch.tensor(coeff_neg_commands, device=self.device)
        self.coeff_pos_commands=torch.tensor(coeff_pos_commands, device=self.device)

        self.interpolate_on_field_data()


    def update(self, thruster_forces_before_dynamics, dt):
        """thrusters dynamics"""

        alpha = torch.exp(torch.tensor((-dt/self.tau), device=self.device))
        self.cmd_updated[:,:] = self.cmd_updated[:,:]*alpha + (1.0 - alpha)*thruster_forces_before_dynamics

        #debugging
        #print(self.cmd_updated[:,:])

        return self.cmd_updated
    
    def compute_thrusters_constant_force(self):
            """for testing purpose"""
    
            #turn
            self.thrusters[:,0]=400
            self.thrusters[:,3]=-400

            return self.thrusters
    
  
    def interpolate_on_field_data(self):
        """interpolates the data furnished by on-field experiment"""

        self.x_linear_interp = torch.linspace(min(self.commands), max(self.commands), self.numberOfPointsForInterpolation)
        self.y_linear_interp = torch.nn.functional.interpolate(self.interpolationPointsFromRealData.unsqueeze(0).unsqueeze(0), size=self.numberOfPointsForInterpolation, mode='linear', align_corners=False)

        self.y_linear_interp = self.y_linear_interp.squeeze(0).squeeze(0) #back to dim 1

        self.n = len(self.y_linear_interp) -1
                
    
    def get_cmd_interpolated(self, cmd_value):
        """get the corresponding force value in the lookup table of interpolated forces"""

        #cmd_value is size (num_envs,2)
        self.idx_matrix = torch.round(((cmd_value + 1) * self.n/2)).to(torch.long)
        
        self.thruster_forces_before_dynamics = self.y_linear_interp[self.idx_matrix]


    def set_target_force(self, commands):  
        """this function get commands as entry and provide resulting forces"""

        #size (num_envs,2)
        self.get_cmd_interpolated(commands)  #every action step
    
        
    def update_forces(self):

        #size (num_envs,2)
        self.thrusters[:,[0,3]] = self.update(self.thruster_forces_before_dynamics, self.dt) #every simulation step that tracks the target  update_thrusters_forces

        return self.thrusters


    """function below has to be change to fit multi robots training"""

    def command_to_thrusters_force_lsm(self, left_thruster_command, right_thruster_command):
         
        """This function implement the non-linearity of the thrusters according to a command""" 
        
        T_left=0
        T_right=0
    
        n=len(self.coeff_neg_commands)-1

        if left_thruster_command<0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*self.coeff_neg_commands[i]
            T_left+=self.coeff_neg_commands[n]
        elif left_thruster_command>=0:
            for i in range(n):
                T_left+=(left_thruster_command**(n-i))*self.coeff_pos_commands[i]
            T_left+=self.coeff_pos_commands[n]
        
        if right_thruster_command<0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*self.coeff_neg_commands[i]
            T_right+=self.coeff_neg_commands[n]
        elif right_thruster_command>=0:
            for i in range(n):
                T_right+=(right_thruster_command**(n-i))*self.coeff_pos_commands[i]
            T_right+=self.coeff_pos_commands[n]


        self.thrusters[:,0]= self.update(T_left, 0.01)
        self.thrusters[:,3]= self.update(T_right, 0.01)

        return self.thrusters

