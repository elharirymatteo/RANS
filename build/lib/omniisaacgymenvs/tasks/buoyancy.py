from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.prims import RigidPrimView
from omniisaacgymenvs.envs.BuoyancyPhysics.Buoyancy_physics import *
from omniisaacgymenvs.envs.BuoyancyPhysics.ThrusterDynamics import *
from omniisaacgymenvs.envs.BuoyancyPhysics.Hydrodynamics import *
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles
 
from omni.physx.scripts import utils
import numpy as np
import torch

class BuoyancyTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        
        #sim config
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        #rl
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength_s"]

        #sim
        self.dt = self._task_cfg["sim"]["dt"]

        #task specifications
        self._num_observations = 7
        self._num_actions = 2
        
        #physics
        self.gravity=self._task_cfg["sim"]["gravity"][2]
        self.water_density=self._task_cfg["buoy"]["water_density"] # kg/m^3
        self.timeConstant = self._task_cfg["dynamics"]["thrusters"]["timeConstant"]

        #buoyancy
        self.average_buoyancy_force_value = self._task_cfg["dynamics"]["buoyancy"]["average_buoyancy_force_value"]
        self.amplify_torque = self._task_cfg["dynamics"]["buoyancy"]["amplify_torque"]

        #thrusters dynamics
            #interpolation
        self.cmd_lower_range = self._task_cfg["dynamics"]["thrusters"]["cmd_lower_range"]
        self.cmd_upper_range = self._task_cfg["dynamics"]["thrusters"]["cmd_upper_range"]
        self.numberOfPointsForInterpolation = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["numberOfPointsForInterpolation"]
        self.interpolationPointsFromRealData = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealData"]
            #least square methode
        self.neg_cmd_coeff=self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["neg_cmd_coeff"]
        self.pos_cmd_coeff=self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["pos_cmd_coeff"]
            #acceleration
        self.alpha = self._task_cfg["dynamics"]["acceleration"]["alpha"]
        self.last_time = self._task_cfg["dynamics"]["acceleration"]["last_time"]

        #boxes dimension to compute buoyancy forces and torques
        self.box_density=self._task_cfg["buoy"]["material_density"]
        self.box_width=self._task_cfg["buoy"]["box_width"]
        self.box_large=self._task_cfg["buoy"]["box_large"]
        self.box_high=self._task_cfg["buoy"]["box_high"]
        self.box_volume=self.box_width*self.box_large*self.box_high
        self.box_mass=self._task_cfg["buoy"]["mass"]
        self.half_box_size=self.box_high/2

        #damping constants
        self.squared_drag_coefficients = self._task_cfg["dynamics"]["damping"]["squared_drag_coefficients"]
        self.linear_damping = self._task_cfg["dynamics"]["damping"]["linear_damping"]
        self.quadratic_damping = self._task_cfg["dynamics"]["damping"]["quadratic_damping"]
        self.linear_damping_forward_speed = self._task_cfg["dynamics"]["damping"]["linear_damping_forward_speed"]
        self.offset_linear_damping = self._task_cfg["dynamics"]["damping"]["offset_linear_damping"]
        self.offset_lin_forward_damping_speed = self._task_cfg["dynamics"]["damping"]["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = self._task_cfg["dynamics"]["damping"]["offset_nonlin_damping"]
        self.scaling_damping = self._task_cfg["dynamics"]["damping"]["scaling_damping"]
        self.offset_added_mass = self._task_cfg["dynamics"]["damping"]["offset_added_mass"]
        self.scaling_added_mass = self._task_cfg["dynamics"]["damping"]["scaling_added_mass"]

        #for testing and debugging
        self.thruster_debugging_counter = 0
        
        RLTask.__init__(self, name=name, env=env)

        #positions constants
        self.left_thruster_position = torch.tensor(self._task_cfg["box"]["left_thruster_position"])
        self.right_thruster_position = torch.tensor(self._task_cfg["box"]["right_thruster_position"])

        #others positions constants that need to be GPU 
        self.boxes_initial_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.boxes_initial_rot = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.boxes_initial_pos[:, 2] = self.box_high/2
        self.boxes_initial_rot[:,0]=0.924    # y 45° rotation : 0.924
        self.boxes_initial_rot[:,2]=0.383    # y 45° rotation : 0.383
        self.boxes_initial_rot[:,1]=0.383


        #volume submerged
        self.high_submerged=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self.submerged_volume=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)

        #forces to be applied
        self.archimedes=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.drag=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.thrusters=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        #obs variables
        self.root_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.root_quats = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.root_velocities = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        ##some tests for the thrusters

        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn = torch.tensor([-1.0, 0.5], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward= - self.forward

        return

    def set_up_scene(self, scene) -> None:
        """create physics objects and set up the scene"""

        self.get_box()
        self.get_buoyancy()
        RLTask.set_up_scene(self, scene)
        self._boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/box/body", name="box_view", reset_xform_properties=False)
        self._thrusters_left= RigidPrimView(prim_paths_expr="/World/envs/.*/box/left_thruster", name="left_thruster_view", reset_xform_properties=False)
        self._thrusters_right= RigidPrimView(prim_paths_expr="/World/envs/.*/box/right_thruster", name="right_thruster_view", reset_xform_properties=False)
        scene.add(self._boxes)
        scene.add(self._thrusters_left)
        scene.add(self._thrusters_right)
        return


    def get_buoyancy(self):
        """create physics"""
        
        self.buoyancy_physics=BuoyantObject(self.num_envs, self._device, self.water_density, self.gravity, self.box_width/2, self.box_large/2, self.average_buoyancy_force_value, self.amplify_torque)
        self.thrusters_dynamics=DynamicsFirstOrder(self.num_envs, self._device, self.timeConstant, self.dt,self.numberOfPointsForInterpolation, self.interpolationPointsFromRealData, self.neg_cmd_coeff, self.pos_cmd_coeff, self.cmd_lower_range, self.cmd_upper_range )
        self.hydrodynamics=HydrodynamicsObject(self.num_envs, self._device, self.squared_drag_coefficients, self.linear_damping, self.quadratic_damping, self.linear_damping_forward_speed, self.offset_linear_damping, self.offset_lin_forward_damping_speed, self.offset_nonlin_damping, self.scaling_damping, self.offset_added_mass, self.scaling_added_mass, self.alpha,self.last_time )

    def get_box(self):
        """add to stage the usd file"""

        box_usd_path="/home/axelcoulon/projects/assets/box_thrusters.usd"
        box_prim_path=self.default_zero_env_path + "/box"
        add_reference_to_stage(prim_path=box_prim_path, usd_path=box_usd_path, prim_type="Xform")

    
    def update_state(self) -> None:
        """
        Updates the state of the system."""

        # Collects the position and orientation of the platform
        self.root_pos[:,:], self.root_quats[:,:] = self._boxes.get_world_poses(clone=False)
        # Collects the velocity of the platform
        self.root_velocities[:,:] = self._boxes.get_velocities(clone=True)

        #get euler angles       
        self.euler_angles = self.get_euler_angles(self.root_quats) #rpy roll pitch yaws

        #body underwater
        self.high_submerged[:]=torch.clamp(self.half_box_size-self.root_pos[:,2], 0, self.box_high)
        self.submerged_volume[:]= torch.clamp(self.high_submerged * self.box_width * self.box_large, 0, self.box_volume)
        self.box_is_under_water = torch.where(self.high_submerged[:] > 0,1.0,0.0 ).unsqueeze(0)

        # Dump to state
        self.current_state = {"position":self.root_pos[:,:3], "orientation":self.root_quats[:,3:], "velocities": self.root_velocities[:,:]}

    def get_observations(self) -> dict:
        """will be implemented after physics"""    

        self.update_state()

        self.obs_buf[..., 0:3] = self.root_pos
        self.obs_buf[..., 3:7] = self.root_quats

        observations = {
            self._boxes.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def get_euler_angles(self, quaternions):
        """quaternions to euler"""

        angles = np.zeros((self.num_envs,3), dtype=float)
        for i in range(self._num_envs):
            angles[i,:]=quat_to_euler_angles(quaternions[i,:])
        return torch.tensor(angles).to(self._device)


    def pre_physics_step(self, actions) -> None:
        """where forces are applied"""

        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.clone().to(self._device)
        indices = torch.arange(self._boxes.count, dtype=torch.int64, device=self._device)

        ###thrusters

        if self.thruster_debugging_counter < 200 :
            self.thrusters_dynamics.set_target_force(self.forward) 
        
        if self.thruster_debugging_counter > 200 and self.thruster_debugging_counter < 600 :
            self.thrusters_dynamics.set_target_force(self.turn) 
        
        if self.thruster_debugging_counter > 600 :
            self.thrusters_dynamics.set_target_force(self.backward) 

        ###apply all the forces 
        self.apply_forces()

        """Printing debugging"""
        print("buoyancy force: ", self.archimedes[0,:3])
        print("buoyancy torques: ", self.archimedes[0,3:])
        print("thrusters: ", self.thrusters[0,:])
        print("drag linear: ", self.drag[0,:3])
        print("drag rotations: ", self.drag[0,3:])
        print("")


    def apply_forces(self):

        
            ###archimedes
        self.archimedes[:,:]=self.buoyancy_physics.compute_archimedes_metacentric_local(self.submerged_volume, self.euler_angles, self.root_quats) * self.box_is_under_water[:,:].mT
            ###drag
        self.drag[:,:]=self.hydrodynamics.ComputeHydrodynamicsEffects(0.01, self.root_quats, self.root_velocities[:,:]) * self.box_is_under_water[:,:].mT
         
        self.thrusters[:,:] = self.thrusters_dynamics.update_forces()
        self.thrusters[:,:] *= self.box_is_under_water.mT
        
        self.thruster_debugging_counter+=1

        self._boxes.apply_forces_and_torques_at_pos(forces=self.archimedes[:,:3] + self.drag[:,:3] , torques=self.archimedes[:,3:] + self.drag[:,3:], is_global=False)
        self._thrusters_left.apply_forces_and_torques_at_pos(self.thrusters[:,:3],positions=self.left_thruster_position,  is_global=False)
        self._thrusters_right.apply_forces_and_torques_at_pos(self.thrusters[:,3:], positions= self.right_thruster_position, is_global=False)
    
    
    def post_reset(self):
        """reset first time before first episode"""

        # randomize all envs
        indices = torch.arange(self._boxes.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.update_state()
       
    def reset_boxes(self, env_ids):
        """reset boxes positions"""

        num_sets = len(env_ids)
        envs_long = env_ids.long()

        # shift the target up so it visually aligns better
        box_pos = self.boxes_initial_pos[envs_long] + self._env_pos[envs_long]
        #box_pos[:,2]=torch.rand(num_sets, device=self._device) + 1.0

        self._boxes.set_world_poses(box_pos[:, 0:3], self.boxes_initial_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        """reset all idx"""

        self.reset_boxes(env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        """reward function"""

        self.rew_buf[:] = 0.0

    def is_done(self) -> None:
        """reset buffer"""

        #no resets for now

        """Flags the environnments in which the episode should end."""
        #resets = torch.where(self.progress_buf >= self._max_episode_length - 1, 1.0, self.reset_buf.double())
        #self.reset_buf[:] = resets
